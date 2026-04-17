#!/usr/bin/env python3
"""
#6: Loss decomposition (Thesis Test 8).

For each Gram direction v_k, compute:
  - G_j^train = <v_j, ∇L_train>  (gradient projection onto v_j)
  - G_j^val = <v_j, ∇L_val>
  - α_j = stability coefficient (eigenvector persistence between half-windows)
  - Predicted importance: α_j * G_j^train * G_j^val
  - Actual importance: loss change when removing direction v_j

Compare predicted vs actual ranking.
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
from scipy.stats import pearsonr, spearmanr

from dyck.grok_sweep import (
    DyckTransformerLM, VOCAB_SIZE, build_depth_dataset, split_dataset,
    sample_batch, masked_ce_loss, masked_accuracy,
)
from spectral.gram_edge_functional_modes import (
    get_attn_param_vector, get_attn_param_keys, compute_gram_svd,
)
from spectral.edge_ablation import build_model, ablate_svd_directions

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"
GRAM_WINDOW = 5


def compute_gradient_projections(model, state_dict, cfg, Vh, param_keys,
                                  data_x, data_y):
    """Compute gradient projections G_j = <v_j, ∇L> for each direction."""
    model.load_state_dict(state_dict)
    model.train()
    model.zero_grad()

    logits = model(data_x)
    loss = masked_ce_loss(logits, data_y)
    loss.backward()

    # Collect gradients for attention params only
    grad_vec = []
    for key in sorted(state_dict.keys()):
        if any(k in key for k in ['in_proj_weight', 'out_proj.weight']):
            param = dict(model.named_parameters())[key]
            if param.grad is not None:
                grad_vec.append(param.grad.detach().cpu().reshape(-1).float())
            else:
                grad_vec.append(torch.zeros(state_dict[key].numel()))
    grad_vec = torch.cat(grad_vec).numpy()

    # Project onto each v_j
    projections = []
    for k in range(min(Vh.shape[0], 6)):
        projections.append(float(np.dot(grad_vec, Vh[k])))

    return projections, float(loss.item())


def compute_stability_coefficient(snapshots, center_idx, window):
    """Compute α_j by comparing eigenvectors between first and second half of window."""
    start = max(1, center_idx - window + 1)
    end = min(len(snapshots) - 1, center_idx)
    n_deltas = end - start + 1
    if n_deltas < 4:  # need at least 4 for 2+2 split
        return None

    # All deltas
    deltas = []
    for i in range(start, end + 1):
        theta_prev = get_attn_param_vector(snapshots[i-1]["state_dict"]).numpy()
        theta_curr = get_attn_param_vector(snapshots[i]["state_dict"]).numpy()
        deltas.append(theta_curr - theta_prev)

    # Split into two halves
    mid = n_deltas // 2
    X1 = np.stack(deltas[:mid])
    X2 = np.stack(deltas[mid:])

    _, _, Vh1 = np.linalg.svd(X1, full_matrices=False)
    _, _, Vh2 = np.linalg.svd(X2, full_matrices=False)

    # α_j = |<v_j^(1), v_j^(2)>| for each direction
    n_dirs = min(Vh1.shape[0], Vh2.shape[0], 6)
    alphas = []
    for j in range(n_dirs):
        alpha = abs(np.dot(Vh1[j], Vh2[j]))
        alphas.append(float(alpha))

    return alphas


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # Data
    X_all, Y_all = build_depth_dataset(n_seqs=5050, max_pairs=12, ctx_len=24, seed=0)
    frac = 50 / 5050
    train_x, train_y, test_x, test_y = split_dataset(X_all, Y_all, frac_train=frac, seed=0)

    phase_indices = {"pre_grok": 3, "at_grok": 6, "post_grok": 15, "late": 40}
    phases = list(phase_indices.keys())

    all_results = {}

    for tag, ckpt_name in [("grok", "dyck_grok_fourier.pt"), ("memo", "dyck_memo_fourier.pt")]:
        ckpt = torch.load(CKPT_DIR / ckpt_name, weights_only=False)
        snapshots = ckpt["snapshots"]
        cfg = ckpt["cfg"]
        param_keys = get_attn_param_keys(snapshots[0]["state_dict"])
        model = build_model(cfg)

        print(f"\n{'='*60}")
        print(f"Loss decomposition: {tag}")
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

            # Gradient projections
            G_train, train_loss = compute_gradient_projections(
                model, state_dict, cfg, Vh, param_keys, train_x, train_y)
            G_val, val_loss = compute_gradient_projections(
                model, state_dict, cfg, Vh, param_keys, test_x[:300], test_y[:300])

            # Stability coefficients
            alphas = compute_stability_coefficient(snapshots, center_idx, GRAM_WINDOW)
            if alphas is None:
                alphas = [1.0] * len(G_train)

            # Predicted importance: α_j * G_j^train * G_j^val
            n_dirs = min(len(G_train), len(G_val), len(alphas), len(S))
            predicted = [alphas[j] * G_train[j] * G_val[j] for j in range(n_dirs)]

            # Actual importance: loss change when removing direction
            actual_loss_change = []
            actual_acc_change = []
            model.load_state_dict(state_dict)
            model.eval()
            with torch.no_grad():
                base_loss = masked_ce_loss(model(test_x[:300]), test_y[:300]).item()
                base_acc = masked_accuracy(model(test_x[:300]), test_y[:300])

            for j in range(n_dirs):
                sd_abl = ablate_svd_directions(state_dict, param_keys, Vh, [j])
                model.load_state_dict(sd_abl)
                model.eval()
                with torch.no_grad():
                    abl_loss = masked_ce_loss(model(test_x[:300]), test_y[:300]).item()
                    abl_acc = masked_accuracy(model(test_x[:300]), test_y[:300])
                actual_loss_change.append(abl_loss - base_loss)
                actual_acc_change.append(abl_acc - base_acc)

            # Correlation: predicted vs actual
            if len(predicted) >= 3:
                r_pearson, p_pearson = pearsonr(np.abs(predicted), np.abs(actual_loss_change))
                r_spearman, p_spearman = spearmanr(np.abs(predicted), np.abs(actual_loss_change))
            else:
                r_pearson = r_spearman = 0
                p_pearson = p_spearman = 1

            print(f"    α: {[f'{a:.3f}' for a in alphas[:n_dirs]]}")
            print(f"    G_train: {[f'{g:.4f}' for g in G_train[:n_dirs]]}")
            print(f"    G_val: {[f'{g:.4f}' for g in G_val[:n_dirs]]}")
            print(f"    Predicted importance: {[f'{p:.6f}' for p in predicted]}")
            print(f"    Actual Δloss: {[f'{a:+.4f}' for a in actual_loss_change]}")
            print(f"    Actual Δacc:  {[f'{a:+.3f}' for a in actual_acc_change]}")
            print(f"    Pearson r(|pred|,|actual|) = {r_pearson:.3f} (p={p_pearson:.3f})")
            print(f"    Spearman ρ = {r_spearman:.3f}")

            all_results[tag][phase_name] = {
                "step": step,
                "alphas": alphas[:n_dirs],
                "G_train": G_train[:n_dirs],
                "G_val": G_val[:n_dirs],
                "predicted": predicted,
                "actual_loss_change": actual_loss_change,
                "actual_acc_change": actual_acc_change,
                "pearson_r": float(r_pearson),
                "spearman_r": float(r_spearman),
                "sigma": S[:n_dirs].tolist(),
            }

    # ── Plot 1: Predicted vs actual importance ──
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Loss Decomposition: Predicted vs Actual Direction Importance", fontsize=14)

    for row, tag in enumerate(["grok", "memo"]):
        for col, phase in enumerate(phases):
            ax = axes[row, col]
            r = all_results[tag].get(phase, {})
            if not r:
                continue
            pred = np.abs(r["predicted"])
            actual = np.abs(r["actual_loss_change"])
            n = len(pred)

            for j in range(n):
                color = "steelblue" if j < 2 else "coral"
                ax.scatter(pred[j], actual[j], c=color, s=80, zorder=5,
                          edgecolors='black', linewidth=0.5)
                ax.annotate(f"v{j+1}", (pred[j], actual[j]), fontsize=8,
                           ha='left', va='bottom')

            ax.set_xlabel("|α·G_train·G_val|")
            ax.set_ylabel("|Δloss|")
            rp = r.get("pearson_r", 0)
            ax.set_title(f"{tag} {phase} (r={rp:.2f})", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "loss_decomposition_pred_vs_actual.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Stability coefficients across phases ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stability Coefficient α_j Across Phases", fontsize=14)

    for i, tag in enumerate(["grok", "memo"]):
        ax = axes[i]
        for k, color in [(0, "steelblue"), (1, "royalblue"), (2, "coral"), (3, "lightsalmon")]:
            vals = []
            for p in phases:
                r = all_results[tag].get(p, {})
                alphas = r.get("alphas", [])
                vals.append(alphas[k] if k < len(alphas) else 0)
            label = "edge" if k < 2 else "bulk"
            ax.plot(range(len(phases)), vals, 'o-', color=color,
                    label=f"v{k+1} ({label})", markersize=6)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=8)
        ax.set_ylabel("α_j (stability)")
        ax.set_title(f"{tag}")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "stability_coefficients.png", dpi=150)
    plt.close(fig)

    # Save
    torch.save(all_results, FIG_DIR / "loss_decomposition_results.pt")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Loss Decomposition (Thesis Test 8)")
    print("="*70)
    for tag in ["grok", "memo"]:
        print(f"\n  {tag}:")
        for phase in phases:
            r = all_results[tag].get(phase, {})
            if not r:
                continue
            print(f"    {phase}: pearson={r.get('pearson_r', 0):.3f}, "
                  f"spearman={r.get('spearman_r', 0):.3f}, "
                  f"α={[f'{a:.2f}' for a in r.get('alphas', [])]}")


if __name__ == "__main__":
    main()
