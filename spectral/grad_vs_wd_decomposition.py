#!/usr/bin/env python3
"""
Decompose weight updates into gradient vs weight-decay components.

AdamW update per step:  θ_{t+1} = θ_t * (1 - lr*wd) - lr * adam_grad_t
So over N steps between checkpoints:
  Δθ_total = θ_{t+N} - θ_t
  Δθ_wd ≈ -lr * wd * N * (θ_t + θ_{t+N})/2   (trapezoidal approx)
  Δθ_grad = Δθ_total - Δθ_wd                    (by subtraction)

For each Gram direction v_k, measure:
  - fraction of update energy from gradient vs weight decay
  - functional perturbation effect of each component separately
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from dyck.grok_sweep import (
    DyckTransformerLM, VOCAB_SIZE, build_depth_dataset, split_dataset,
    masked_ce_loss, masked_accuracy,
)

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
SCAN_CKPT_DIR = Path(__file__).resolve().parent / "fourier_scan_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"
SCAN_FIG_DIR = Path(__file__).resolve().parent / "fourier_scan_plots"

GRAM_WINDOW = 5


def get_attn_param_vector(state_dict):
    """Flatten all attention params into one vector."""
    parts = []
    for key in sorted(state_dict.keys()):
        if any(k in key for k in ['in_proj_weight', 'out_proj.weight']):
            parts.append(state_dict[key].reshape(-1).float())
    return torch.cat(parts)


def decompose_updates(snapshots, lr, wd, steps_per_interval):
    """Decompose consecutive checkpoint deltas into grad and WD components.

    Returns list of dicts with delta_total, delta_grad, delta_wd for each interval.
    """
    decomposed = []
    for i in range(1, len(snapshots)):
        theta_prev = get_attn_param_vector(snapshots[i-1]["state_dict"]).numpy()
        theta_curr = get_attn_param_vector(snapshots[i]["state_dict"]).numpy()

        delta_total = theta_curr - theta_prev

        # Weight decay component (trapezoidal approximation)
        # Over N steps: Δθ_wd ≈ -lr * wd * N * (θ_prev + θ_curr) / 2
        N = snapshots[i]["step"] - snapshots[i-1]["step"]
        delta_wd = -lr * wd * N * (theta_prev + theta_curr) / 2.0

        # Gradient component (by subtraction)
        delta_grad = delta_total - delta_wd

        decomposed.append({
            "step": snapshots[i]["step"],
            "delta_total": delta_total,
            "delta_grad": delta_grad,
            "delta_wd": delta_wd,
            "n_steps": N,
        })
    return decomposed


def compute_gram_svd_from_deltas(deltas_list, center_idx, window):
    """Compute Gram SVD from pre-computed deltas."""
    start = max(0, center_idx - window + 1)
    end = min(len(deltas_list) - 1, center_idx)
    if end - start + 1 < 2:
        return None

    X = np.stack([deltas_list[i]["delta_total"] for i in range(start, end + 1)])
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    return {"singular_values": S, "Vh": Vh}


def project_decomposition(deltas_list, center_idx, window, Vh, topk=5):
    """Project grad and WD components onto Gram directions.

    Returns per-direction energy fractions.
    """
    start = max(0, center_idx - window + 1)
    end = min(len(deltas_list) - 1, center_idx)
    if end - start + 1 < 2:
        return None

    results = {}
    for k in range(min(topk, Vh.shape[0])):
        v_k = Vh[k]  # [p]

        grad_projections = []
        wd_projections = []
        total_projections = []

        for i in range(start, end + 1):
            d = deltas_list[i]
            grad_projections.append(np.dot(d["delta_grad"], v_k))
            wd_projections.append(np.dot(d["delta_wd"], v_k))
            total_projections.append(np.dot(d["delta_total"], v_k))

        # Energy along this direction
        grad_energy = np.mean(np.array(grad_projections) ** 2)
        wd_energy = np.mean(np.array(wd_projections) ** 2)
        total_energy = np.mean(np.array(total_projections) ** 2)

        # Signed mean (direction of push)
        grad_mean = np.mean(grad_projections)
        wd_mean = np.mean(wd_projections)

        # Fraction
        denom = grad_energy + wd_energy
        grad_frac = grad_energy / denom if denom > 0 else 0.5
        wd_frac = wd_energy / denom if denom > 0 else 0.5

        # Alignment: are grad and WD pushing same or opposite direction?
        alignment = np.mean(np.array(grad_projections) * np.array(wd_projections))
        align_sign = "aligned" if alignment > 0 else "opposing"

        results[f"v{k+1}"] = {
            "grad_energy": float(grad_energy),
            "wd_energy": float(wd_energy),
            "total_energy": float(total_energy),
            "grad_frac": float(grad_frac),
            "wd_frac": float(wd_frac),
            "grad_mean": float(grad_mean),
            "wd_mean": float(wd_mean),
            "alignment": align_sign,
            "align_value": float(alignment),
        }

    return results


def functional_perturbation_of_component(model, state_dict, direction_vec,
                                          component_delta, param_keys,
                                          test_x, test_y, eps_scale=0.005):
    """Measure functional effect of perturbing along a direction
    using only the gradient or WD component of a delta.

    Project component_delta onto direction_vec, then perturb model.
    Returns change in loss and accuracy.
    """
    # How much of the component lies along this direction
    projection = np.dot(component_delta, direction_vec)
    if abs(projection) < 1e-12:
        return {"loss_change": 0.0, "acc_change": 0.0, "projection": 0.0}

    # Perturb model by projection * v_k
    theta_attn = get_attn_param_vector(state_dict)
    eps = eps_scale * torch.norm(theta_attn).item()

    perturbed_sd = {k: v.clone() for k, v in state_dict.items()}
    offset = 0
    for key, numel in param_keys:
        chunk = direction_vec[offset:offset + numel]
        perturbed_sd[key] = perturbed_sd[key] + eps * np.sign(projection) * torch.tensor(
            chunk, dtype=perturbed_sd[key].dtype
        ).reshape(perturbed_sd[key].shape)
        offset += numel

    # Compute loss difference
    model.eval()
    with torch.no_grad():
        logits_base = model(test_x)
        loss_base = masked_ce_loss(logits_base, test_y).item()
        acc_base = masked_accuracy(logits_base, test_y)

    # Load perturbed
    model.load_state_dict(perturbed_sd)
    model.eval()
    with torch.no_grad():
        logits_pert = model(test_x)
        loss_pert = masked_ce_loss(logits_pert, test_y).item()
        acc_pert = masked_accuracy(logits_pert, test_y)

    # Restore original
    model.load_state_dict(state_dict)

    return {
        "loss_change": loss_pert - loss_base,
        "acc_change": acc_pert - acc_base,
        "projection": float(projection),
    }


def get_param_keys(state_dict):
    keys = []
    for key in sorted(state_dict.keys()):
        if any(k in key for k in ['in_proj_weight', 'out_proj.weight']):
            keys.append((key, state_dict[key].numel()))
    return keys


# ═══════════════════════════════════════════════════════════════════════
# DYCK Analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_dyck():
    print("\n" + "="*70)
    print("DYCK: Gradient vs Weight Decay Decomposition")
    print("="*70)

    # Load test data
    X_all, Y_all = build_depth_dataset(n_seqs=5050, max_pairs=12, ctx_len=24, seed=0)
    _, _, test_x, test_y = split_dataset(X_all, Y_all, frac_train=50/5050, seed=0)
    test_x_small = test_x[:200]
    test_y_small = test_y[:200]

    phase_indices = {"pre_grok": 2, "at_grok": 5, "post_grok": 14, "late": 39}

    all_results = {}

    for tag, ckpt_name in [("grok", "dyck_grok_fourier.pt"), ("memo", "dyck_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        ckpt = torch.load(ckpt_path, weights_only=False)
        snapshots = ckpt["snapshots"]
        cfg = ckpt["cfg"]
        lr = cfg["LR"]
        wd = cfg["WEIGHT_DECAY"]

        print(f"\n  {tag}: lr={lr}, wd={wd}, {len(snapshots)} snapshots")

        # Decompose all updates
        decomposed = decompose_updates(snapshots, lr, wd, steps_per_interval=100)
        print(f"  {len(decomposed)} intervals decomposed")

        # Build model
        model = DyckTransformerLM(
            vocab_size=VOCAB_SIZE,
            ctx_len=max(cfg["CTX_LEN"], cfg["CTX_LEN_OOD"]),
            d_model=cfg["D_MODEL"], n_layers=cfg["N_LAYERS"],
            n_heads=cfg["N_HEADS"], d_ff=cfg["D_FF"],
            dropout=cfg["DROPOUT"], n_classes=cfg["N_CLASSES"],
        )
        param_keys = get_param_keys(snapshots[0]["state_dict"])

        all_results[tag] = {}

        for phase_name, center_idx in phase_indices.items():
            if center_idx >= len(decomposed):
                continue

            center_step = decomposed[center_idx]["step"]
            print(f"\n  {phase_name} (step {center_step}):")

            # Gram SVD
            gram = compute_gram_svd_from_deltas(decomposed, center_idx, GRAM_WINDOW)
            if gram is None:
                continue

            S = gram["singular_values"]
            Vh = gram["Vh"]
            print(f"    σ = {S[:5].round(4)}")

            # Project decomposition
            proj = project_decomposition(decomposed, center_idx, GRAM_WINDOW, Vh, topk=5)

            # Functional perturbation of grad vs WD components
            model.load_state_dict(snapshots[center_idx + 1]["state_dict"])
            model.eval()

            # Use the delta at center_idx
            delta = decomposed[center_idx]

            for vname, vdata in proj.items():
                k = int(vname[1]) - 1
                v_k = Vh[k]

                # Functional effect of grad component along v_k
                grad_func = functional_perturbation_of_component(
                    model, snapshots[center_idx + 1]["state_dict"],
                    v_k, delta["delta_grad"], param_keys,
                    test_x_small, test_y_small)

                # Functional effect of WD component along v_k
                wd_func = functional_perturbation_of_component(
                    model, snapshots[center_idx + 1]["state_dict"],
                    v_k, delta["delta_wd"], param_keys,
                    test_x_small, test_y_small)

                vdata["grad_loss_change"] = grad_func["loss_change"]
                vdata["grad_acc_change"] = grad_func["acc_change"]
                vdata["wd_loss_change"] = wd_func["loss_change"]
                vdata["wd_acc_change"] = wd_func["acc_change"]

                label = "EDGE" if k < 2 else "BULK"
                print(f"    {vname} ({label}): grad_frac={vdata['grad_frac']:.3f}, "
                      f"wd_frac={vdata['wd_frac']:.3f}, {vdata['alignment']}")
                print(f"      grad→loss: {grad_func['loss_change']:+.4f}, "
                      f"grad→acc: {grad_func['acc_change']:+.4f}")
                print(f"      wd→loss:   {wd_func['loss_change']:+.4f}, "
                      f"wd→acc:   {wd_func['acc_change']:+.4f}")

            all_results[tag][phase_name] = {
                "step": center_step,
                "singular_values": S.tolist(),
                "directions": proj,
            }

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# SCAN Analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_scan():
    from scan.grok_sweep import ScanTransformer, masked_ce_loss as scan_ce, masked_accuracy as scan_acc

    print("\n" + "="*70)
    print("SCAN: Gradient vs Weight Decay Decomposition")
    print("="*70)

    phase_indices = {"early": 1, "pre_grok": 4, "at_grok": 9, "post_grok": 19}

    all_results = {}

    for tag, ckpt_name in [("grok", "scan_grok_fourier.pt"), ("memo", "scan_memo_fourier.pt")]:
        ckpt_path = SCAN_CKPT_DIR / ckpt_name
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, weights_only=False)
        snapshots = ckpt["snapshots"]
        cfg = ckpt["cfg"]
        lr = cfg["LR"]
        wd = cfg["WEIGHT_DECAY"]

        print(f"\n  {tag}: lr={lr}, wd={wd}, {len(snapshots)} snapshots")

        decomposed = decompose_updates(snapshots, lr, wd, steps_per_interval=500)
        print(f"  {len(decomposed)} intervals decomposed")

        all_results[tag] = {}

        for phase_name, center_idx in phase_indices.items():
            if center_idx >= len(decomposed):
                continue

            center_step = decomposed[center_idx]["step"]
            print(f"\n  {phase_name} (step {center_step}):")

            gram = compute_gram_svd_from_deltas(decomposed, center_idx, GRAM_WINDOW)
            if gram is None:
                continue

            S = gram["singular_values"]
            Vh = gram["Vh"]
            print(f"    σ = {S[:5].round(4)}")

            proj = project_decomposition(decomposed, center_idx, GRAM_WINDOW, Vh, topk=5)

            for vname, vdata in proj.items():
                label = "EDGE" if vname in ["v1", "v2"] else "BULK"
                print(f"    {vname} ({label}): grad_frac={vdata['grad_frac']:.3f}, "
                      f"wd_frac={vdata['wd_frac']:.3f}, {vdata['alignment']}")

            all_results[tag][phase_name] = {
                "step": center_step,
                "singular_values": S.tolist(),
                "directions": proj,
            }

    return all_results


def make_plots(dyck_results, scan_results):
    """Generate comparison plots."""

    # ── Plot 1: Grad vs WD fraction across phases ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Gradient vs Weight Decay Energy Fraction per Direction", fontsize=14)

    phases_dyck = ["pre_grok", "at_grok", "post_grok", "late"]
    phases_scan = ["early", "pre_grok", "at_grok", "post_grok"]

    for row, (dataset, results, phases, fig_dir) in enumerate([
        ("Dyck", dyck_results, phases_dyck, FIG_DIR),
        ("SCAN", scan_results, phases_scan, SCAN_FIG_DIR),
    ]):
        for col, tag in enumerate(["grok", "memo"]):
            if tag not in results:
                continue
            ax = axes[row, col]

            for k, color in [(0, "steelblue"), (1, "royalblue"), (2, "coral"), (3, "lightsalmon")]:
                grad_fracs = []
                for p in phases:
                    r = results[tag].get(p, {})
                    d = r.get("directions", {}).get(f"v{k+1}", {})
                    grad_fracs.append(d.get("grad_frac", 0.5))

                label_type = "edge" if k < 2 else "bulk"
                ax.plot(range(len(phases)), grad_fracs, 'o-', color=color,
                        label=f"v{k+1} ({label_type})", markersize=6)

            ax.axhline(y=0.5, ls='--', color='gray', alpha=0.5, label='50/50')
            ax.set_xticks(range(len(phases)))
            ax.set_xticklabels(phases, fontsize=8)
            ax.set_ylabel("Gradient fraction")
            ax.set_title(f"{dataset} {tag}")
            ax.legend(fontsize=6)
            ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "grad_vs_wd_fraction.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'grad_vs_wd_fraction.png'}")

    # ── Plot 2: Functional effect of grad vs WD (Dyck only, has perturbation data) ──
    if dyck_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Dyck: Functional Effect of Gradient vs WD Components", fontsize=14)

        for col, metric in enumerate(["loss_change", "acc_change"]):
            for row, tag in enumerate(["grok", "memo"]):
                if tag not in dyck_results:
                    continue
                ax = axes[row, col]

                for k, color in [(0, "steelblue"), (1, "royalblue"), (2, "coral")]:
                    grad_vals = []
                    wd_vals = []
                    for p in phases_dyck:
                        r = dyck_results[tag].get(p, {})
                        d = r.get("directions", {}).get(f"v{k+1}", {})
                        grad_vals.append(d.get(f"grad_{metric}", 0))
                        wd_vals.append(d.get(f"wd_{metric}", 0))

                    x = np.arange(len(phases_dyck))
                    w = 0.12
                    ax.bar(x + k*2*w - 0.15, grad_vals, w, color=color, alpha=0.8,
                           label=f"v{k+1} grad" if row == 0 and col == 0 else "")
                    ax.bar(x + k*2*w - 0.15 + w, wd_vals, w, color=color, alpha=0.4,
                           hatch='//', label=f"v{k+1} WD" if row == 0 and col == 0 else "")

                ax.set_xticks(range(len(phases_dyck)))
                ax.set_xticklabels(phases_dyck, fontsize=8)
                ylabel = "Δloss" if "loss" in metric else "Δacc"
                ax.set_ylabel(ylabel)
                ax.set_title(f"{tag}: {ylabel}")
                ax.axhline(y=0, ls='-', color='gray', alpha=0.3)
                if row == 0 and col == 0:
                    ax.legend(fontsize=6, ncol=2)

        plt.tight_layout()
        fig.savefig(FIG_DIR / "grad_vs_wd_functional_effect.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {FIG_DIR / 'grad_vs_wd_functional_effect.png'}")

    # ── Plot 3: Alignment (grad and WD pushing same or opposite direction) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Grad-WD Alignment Along Gram Directions", fontsize=14)

    for i, (dataset, results, phases) in enumerate([
        ("Dyck", dyck_results, phases_dyck),
        ("SCAN", scan_results, phases_scan),
    ]):
        ax = axes[i]
        for tag, ls in [("grok", "-"), ("memo", "--")]:
            if tag not in results:
                continue
            for k, color in [(0, "steelblue"), (1, "royalblue"), (2, "coral")]:
                vals = []
                for p in phases:
                    r = results[tag].get(p, {})
                    d = r.get("directions", {}).get(f"v{k+1}", {})
                    vals.append(d.get("align_value", 0))
                label_type = "edge" if k < 2 else "bulk"
                ax.plot(range(len(phases)), vals, f'o{ls}', color=color,
                        label=f"{tag} v{k+1}" if ls == "-" else "", markersize=5, alpha=0.7)

        ax.axhline(y=0, ls='--', color='gray', alpha=0.5)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=8)
        ax.set_ylabel("⟨grad·WD⟩ along v_k (>0=aligned, <0=opposing)")
        ax.set_title(dataset)
        ax.legend(fontsize=6)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "grad_wd_alignment.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'grad_wd_alignment.png'}")


def main():
    dyck_results = analyze_dyck()
    scan_results = analyze_scan()
    make_plots(dyck_results, scan_results)

    # Save
    torch.save({"dyck": dyck_results, "scan": scan_results},
               FIG_DIR / "grad_vs_wd_results.pt")

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY: Gradient vs Weight Decay Decomposition")
    print("="*70)

    for dataset, results, phases in [
        ("DYCK", dyck_results, ["pre_grok", "at_grok", "post_grok", "late"]),
        ("SCAN", scan_results, ["early", "pre_grok", "at_grok", "post_grok"]),
    ]:
        for tag in ["grok", "memo"]:
            if tag not in results:
                continue
            print(f"\n  {dataset} {tag}:")
            for phase in phases:
                r = results[tag].get(phase, {})
                if not r:
                    continue
                print(f"    {phase} (step {r.get('step', '?')}):")
                for vname in ["v1", "v2", "v3"]:
                    d = r.get("directions", {}).get(vname, {})
                    if d:
                        label = "EDGE" if vname in ["v1", "v2"] else "BULK"
                        print(f"      {vname} ({label}): grad={d.get('grad_frac',0):.1%} "
                              f"wd={d.get('wd_frac',0):.1%} [{d.get('alignment','?')}]")


if __name__ == "__main__":
    main()
