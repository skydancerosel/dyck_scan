#!/usr/bin/env python3
"""
#8: Edge/bulk ablation at inference.

Remove contributions from edge or bulk directions and measure causal effect:
  - Project out top-k singular components from W_Q or W_K
  - Measure change in task accuracy, loss, depth probes

#6: Path-norm vs function change per direction (combined here).
  - For each v_k: ||Δθ|| (param displacement) vs Δloss, Δacc (function change)

#3: Hessian curvature via finite-difference Hessian-vector products.
  - v_k^T H v_k ≈ (L(θ+εv) + L(θ-εv) - 2L(θ)) / ε²
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

from dyck.grok_sweep import (
    DyckTransformerLM, VOCAB_SIZE, build_depth_dataset, split_dataset,
    masked_ce_loss, masked_accuracy, eval_on_dataset,
)
from spectral.gram_edge_functional_modes import get_attn_param_vector, get_attn_param_keys, compute_gram_svd

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"
GRAM_WINDOW = 5


def build_model(cfg):
    return DyckTransformerLM(
        vocab_size=VOCAB_SIZE,
        ctx_len=max(cfg["CTX_LEN"], cfg["CTX_LEN_OOD"]),
        d_model=cfg["D_MODEL"], n_layers=cfg["N_LAYERS"],
        n_heads=cfg["N_HEADS"], d_ff=cfg["D_FF"],
        dropout=cfg["DROPOUT"], n_classes=cfg["N_CLASSES"],
    )


def ablate_svd_directions(state_dict, param_keys, Vh, indices_to_remove, scale=1.0):
    """Remove (project out) specified SVD directions from attention params.

    Modifies state_dict by subtracting projection onto each v_k.
    scale controls how much to remove (1.0 = full ablation).
    """
    ablated_sd = {k: v.clone() for k, v in state_dict.items()}
    theta = get_attn_param_vector(state_dict).numpy()

    # Compute projection to remove
    removal = np.zeros_like(theta)
    for k in indices_to_remove:
        v_k = Vh[k]
        proj = np.dot(theta, v_k)
        removal += scale * proj * v_k

    # Apply to state dict
    offset = 0
    for key, numel in param_keys:
        chunk = removal[offset:offset + numel]
        ablated_sd[key] = ablated_sd[key] - torch.tensor(
            chunk, dtype=ablated_sd[key].dtype
        ).reshape(ablated_sd[key].shape)
        offset += numel

    return ablated_sd


def hessian_curvature(model, cfg, state_dict, param_keys, v_k, test_x, test_y, eps=0.001):
    """Compute directional Hessian curvature: v^T H v ≈ (L(θ+εv) + L(θ-εv) - 2L(θ)) / ε²"""
    def eval_loss(sd):
        m = build_model(cfg)
        m.load_state_dict(sd)
        m.eval()
        with torch.no_grad():
            logits = m(test_x)
            return masked_ce_loss(logits, test_y).item()

    # Scale eps by param norm
    theta_norm = get_attn_param_vector(state_dict).norm().item()
    eps_scaled = eps * theta_norm

    # L(θ)
    L0 = eval_loss(state_dict)

    # L(θ + εv)
    sd_plus = {k: v.clone() for k, v in state_dict.items()}
    offset = 0
    for key, numel in param_keys:
        chunk = v_k[offset:offset + numel]
        sd_plus[key] = sd_plus[key] + eps_scaled * torch.tensor(chunk, dtype=sd_plus[key].dtype).reshape(sd_plus[key].shape)
        offset += numel
    L_plus = eval_loss(sd_plus)

    # L(θ - εv)
    sd_minus = {k: v.clone() for k, v in state_dict.items()}
    offset = 0
    for key, numel in param_keys:
        chunk = v_k[offset:offset + numel]
        sd_minus[key] = sd_minus[key] - eps_scaled * torch.tensor(chunk, dtype=sd_minus[key].dtype).reshape(sd_minus[key].shape)
        offset += numel
    L_minus = eval_loss(sd_minus)

    curvature = (L_plus + L_minus - 2 * L0) / (eps_scaled ** 2)
    return curvature, L0, L_plus, L_minus


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    X_all, Y_all = build_depth_dataset(n_seqs=5050, max_pairs=12, ctx_len=24, seed=0)
    _, _, test_x, test_y = split_dataset(X_all, Y_all, frac_train=50/5050, seed=0)
    test_x = test_x[:300]
    test_y = test_y[:300]

    phase_indices = {"pre_grok": 2, "at_grok": 5, "post_grok": 14, "late": 39}
    phases = list(phase_indices.keys())

    all_results = {}

    for tag, ckpt_name in [("grok", "dyck_grok_fourier.pt"), ("memo", "dyck_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        ckpt = torch.load(ckpt_path, weights_only=False)
        snapshots = ckpt["snapshots"]
        cfg = ckpt["cfg"]
        param_keys = get_attn_param_keys(snapshots[0]["state_dict"])

        print(f"\n{'='*60}")
        print(f"Ablation + Hessian + Path-norm: {tag}")
        print(f"{'='*60}")

        all_results[tag] = {}

        for phase_name, center_idx in phase_indices.items():
            if center_idx >= len(snapshots):
                continue

            state_dict = snapshots[center_idx]["state_dict"]
            step = snapshots[center_idx]["step"]
            gram = compute_gram_svd(snapshots, center_idx, GRAM_WINDOW)
            if gram is None:
                continue

            S = gram["singular_values"]
            Vh = gram["Vh"]

            print(f"\n  {phase_name} (step {step}):")

            # Base accuracy
            model = build_model(cfg)
            model.load_state_dict(state_dict)
            model.eval()
            with torch.no_grad():
                logits_base = model(test_x)
                base_loss = masked_ce_loss(logits_base, test_y).item()
                base_acc = masked_accuracy(logits_base, test_y)
            print(f"    Base: loss={base_loss:.4f}, acc={base_acc:.3f}")

            phase_results = {"step": step, "base_loss": base_loss, "base_acc": base_acc, "directions": {}}

            n_dirs = min(4, len(S))

            # ── Ablation ──
            # Ablate edge (v1+v2)
            sd_no_edge = ablate_svd_directions(state_dict, param_keys, Vh, [0, 1])
            model.load_state_dict(sd_no_edge)
            model.eval()
            with torch.no_grad():
                logits_ne = model(test_x)
                ne_loss = masked_ce_loss(logits_ne, test_y).item()
                ne_acc = masked_accuracy(logits_ne, test_y)
            print(f"    Ablate edge (v1+v2): loss={ne_loss:.4f}, acc={ne_acc:.3f}, "
                  f"Δacc={ne_acc - base_acc:+.3f}")

            # Ablate bulk (v3+v4)
            if n_dirs >= 4:
                sd_no_bulk = ablate_svd_directions(state_dict, param_keys, Vh, [2, 3])
                model.load_state_dict(sd_no_bulk)
                model.eval()
                with torch.no_grad():
                    logits_nb = model(test_x)
                    nb_loss = masked_ce_loss(logits_nb, test_y).item()
                    nb_acc = masked_accuracy(logits_nb, test_y)
                print(f"    Ablate bulk (v3+v4): loss={nb_loss:.4f}, acc={nb_acc:.3f}, "
                      f"Δacc={nb_acc - base_acc:+.3f}")
            else:
                nb_loss, nb_acc = base_loss, base_acc

            phase_results["ablate_edge"] = {"loss": ne_loss, "acc": ne_acc,
                                             "delta_acc": ne_acc - base_acc}
            phase_results["ablate_bulk"] = {"loss": nb_loss, "acc": nb_acc,
                                             "delta_acc": nb_acc - base_acc}

            # ── Hessian curvature + Path-norm per direction ──
            for k in range(n_dirs):
                v_k = Vh[k]
                label = "EDGE" if k < 2 else "BULK"

                # Hessian
                curv, L0, Lp, Lm = hessian_curvature(
                    model, cfg, state_dict, param_keys, v_k, test_x, test_y)

                # Path-norm: ||Δθ_k|| = σ_k (singular value IS the norm of updates along v_k)
                # Function change: |L(θ+εv) - L(θ)|
                func_change = abs(Lp - L0)

                # Single-direction ablation
                sd_abl_k = ablate_svd_directions(state_dict, param_keys, Vh, [k])
                model.load_state_dict(sd_abl_k)
                model.eval()
                with torch.no_grad():
                    logits_k = model(test_x)
                    abl_loss = masked_ce_loss(logits_k, test_y).item()
                    abl_acc = masked_accuracy(logits_k, test_y)

                print(f"    v{k+1} ({label}): Hessian_curv={curv:.4f}, σ={S[k]:.4f}, "
                      f"func_Δ={func_change:.5f}, ablate_Δacc={abl_acc - base_acc:+.3f}")

                phase_results["directions"][f"v{k+1}"] = {
                    "sigma": float(S[k]),
                    "hessian_curvature": float(curv),
                    "func_change": float(func_change),
                    "ablate_loss": float(abl_loss),
                    "ablate_acc": float(abl_acc),
                    "ablate_delta_acc": float(abl_acc - base_acc),
                }

            all_results[tag][phase_name] = phase_results

    # ══════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════

    # ── Plot 1: Ablation effect ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ablation Effect: Edge vs Bulk Removal", fontsize=14)

    for i, tag in enumerate(["grok", "memo"]):
        ax = axes[i]
        edge_delta = [all_results[tag].get(p, {}).get("ablate_edge", {}).get("delta_acc", 0) for p in phases]
        bulk_delta = [all_results[tag].get(p, {}).get("ablate_bulk", {}).get("delta_acc", 0) for p in phases]
        x = np.arange(len(phases))
        ax.bar(x - 0.15, edge_delta, 0.3, color="steelblue", label="Remove edge")
        ax.bar(x + 0.15, bulk_delta, 0.3, color="coral", label="Remove bulk")
        ax.set_xticks(x)
        ax.set_xticklabels(phases, fontsize=8)
        ax.set_ylabel("Δ accuracy")
        ax.set_title(f"{tag}")
        ax.legend()
        ax.axhline(y=0, ls='-', color='gray', alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "ablation_edge_vs_bulk.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Hessian curvature ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Hessian Directional Curvature: Edge vs Bulk", fontsize=14)

    for i, tag in enumerate(["grok", "memo"]):
        ax = axes[i]
        for k, color in [(0, "steelblue"), (1, "royalblue"), (2, "coral"), (3, "lightsalmon")]:
            vals = [all_results[tag].get(p, {}).get("directions", {}).get(f"v{k+1}", {}).get("hessian_curvature", 0) for p in phases]
            label = "edge" if k < 2 else "bulk"
            ax.plot(range(len(phases)), vals, 'o-', color=color, label=f"v{k+1} ({label})", markersize=6)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=8)
        ax.set_ylabel("v^T H v (curvature)")
        ax.set_title(f"{tag}")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "hessian_curvature_edge_bulk.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Path-norm (σ) vs function change ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Path-norm (σ) vs Function Change: Edge vs Bulk", fontsize=14)

    for i, tag in enumerate(["grok", "memo"]):
        ax = axes[i]
        for phase, marker in zip(phases, ['o', 's', '^', 'D']):
            for k in range(4):
                d = all_results[tag].get(phase, {}).get("directions", {}).get(f"v{k+1}", {})
                if d:
                    color = "steelblue" if k < 2 else "coral"
                    ax.scatter(d["sigma"], d["func_change"], c=color, marker=marker,
                              s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.set_xlabel("σ_k (update magnitude)")
        ax.set_ylabel("|L(θ+εv) - L(θ)| (function change)")
        ax.set_title(f"{tag}")
        # Legend for phases
        for phase, marker in zip(phases, ['o', 's', '^', 'D']):
            ax.scatter([], [], c='gray', marker=marker, label=phase, s=40)
        ax.scatter([], [], c='steelblue', marker='o', label='edge', s=40)
        ax.scatter([], [], c='coral', marker='o', label='bulk', s=40)
        ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "pathnorm_vs_funcchange.png", dpi=150)
    plt.close(fig)

    # Save
    torch.save(all_results, FIG_DIR / "ablation_hessian_results.pt")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Ablation + Hessian + Path-norm")
    print("="*70)
    for tag in ["grok", "memo"]:
        print(f"\n  {tag}:")
        for phase in phases:
            r = all_results[tag].get(phase, {})
            if not r:
                continue
            print(f"    {phase} (step {r.get('step','?')}):")
            ae = r.get("ablate_edge", {})
            ab = r.get("ablate_bulk", {})
            print(f"      Ablate edge: Δacc={ae.get('delta_acc',0):+.3f}")
            print(f"      Ablate bulk: Δacc={ab.get('delta_acc',0):+.3f}")
            for vname in ["v1", "v2", "v3", "v4"]:
                d = r.get("directions", {}).get(vname, {})
                if d:
                    lb = "E" if vname in ["v1", "v2"] else "B"
                    print(f"      {vname}({lb}): H_curv={d.get('hessian_curvature',0):.4f}, "
                          f"σ={d.get('sigma',0):.4f}, Δfunc={d.get('func_change',0):.5f}")


if __name__ == "__main__":
    main()
