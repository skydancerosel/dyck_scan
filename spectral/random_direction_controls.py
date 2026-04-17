#!/usr/bin/env python3
"""
#5: Random direction controls for ablation.

For each phase, compare ablating:
  - Edge directions (v1, v2) from Gram SVD
  - Random directions of the SAME NORM
  - Random directions from the parameter space

This tests whether the edge is special or if any high-norm direction matters.
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
    masked_ce_loss, masked_accuracy,
)
from spectral.gram_edge_functional_modes import (
    get_attn_param_vector, get_attn_param_keys, compute_gram_svd,
)
from spectral.edge_ablation import build_model, ablate_svd_directions

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"
GRAM_WINDOW = 5
N_RANDOM = 20  # number of random ablation trials


def ablate_random_directions(state_dict, param_keys, n_dirs, norm_target, rng):
    """Ablate n_dirs random orthonormal directions with total norm matching edge."""
    p = sum(n for _, n in param_keys)

    # Generate random orthonormal directions
    raw = rng.randn(n_dirs, p)
    Q, _ = np.linalg.qr(raw.T)
    random_dirs = Q[:, :n_dirs].T  # [n_dirs, p]

    # Scale to match norm: edge ablation removes projection of θ onto edge
    theta = get_attn_param_vector(state_dict).numpy()
    removal = np.zeros_like(theta)
    for k in range(n_dirs):
        proj = np.dot(theta, random_dirs[k])
        removal += proj * random_dirs[k]

    # Apply
    ablated_sd = {k: v.clone() for k, v in state_dict.items()}
    offset = 0
    for key, numel in param_keys:
        chunk = removal[offset:offset + numel]
        ablated_sd[key] = ablated_sd[key] - torch.tensor(
            chunk, dtype=ablated_sd[key].dtype).reshape(ablated_sd[key].shape)
        offset += numel

    return ablated_sd


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X_all, Y_all = build_depth_dataset(n_seqs=5050, max_pairs=12, ctx_len=24, seed=0)
    _, _, test_x, test_y = split_dataset(X_all, Y_all, frac_train=50/5050, seed=0)
    test_x = test_x[:300]
    test_y = test_y[:300]

    phase_indices = {"pre_grok": 2, "at_grok": 5, "post_grok": 14, "late": 39}
    phases = list(phase_indices.keys())

    ckpt = torch.load(CKPT_DIR / "dyck_grok_fourier.pt", weights_only=False)
    snapshots = ckpt["snapshots"]
    cfg = ckpt["cfg"]
    param_keys = get_attn_param_keys(snapshots[0]["state_dict"])

    all_results = {}

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
            base_acc = masked_accuracy(model(test_x), test_y)
        print(f"    Base acc: {base_acc:.3f}")

        # Edge ablation (v1+v2)
        sd_edge = ablate_svd_directions(state_dict, param_keys, Vh, [0, 1])
        model.load_state_dict(sd_edge)
        with torch.no_grad():
            edge_acc = masked_accuracy(model(test_x), test_y)
        edge_delta = edge_acc - base_acc
        print(f"    Edge ablation: Δacc = {edge_delta:+.3f}")

        # Random ablations (matched: 2 directions)
        random_deltas = []
        for trial in range(N_RANDOM):
            rng = np.random.RandomState(trial + 1000)
            sd_rand = ablate_random_directions(state_dict, param_keys, 2, None, rng)
            model.load_state_dict(sd_rand)
            with torch.no_grad():
                rand_acc = masked_accuracy(model(test_x), test_y)
            random_deltas.append(rand_acc - base_acc)

        random_deltas = np.array(random_deltas)
        print(f"    Random ablation (n={N_RANDOM}): mean Δacc = {random_deltas.mean():+.3f} "
              f"± {random_deltas.std():.3f}")
        print(f"    Edge is {abs(edge_delta) / (abs(random_deltas.mean()) + 1e-8):.1f}x worse than random mean")

        # Single-direction ablations for finer comparison
        v1_sd = ablate_svd_directions(state_dict, param_keys, Vh, [0])
        model.load_state_dict(v1_sd)
        with torch.no_grad():
            v1_acc = masked_accuracy(model(test_x), test_y)

        # Single random direction
        single_random = []
        for trial in range(N_RANDOM):
            rng = np.random.RandomState(trial + 2000)
            sd_r1 = ablate_random_directions(state_dict, param_keys, 1, None, rng)
            model.load_state_dict(sd_r1)
            with torch.no_grad():
                r1_acc = masked_accuracy(model(test_x), test_y)
            single_random.append(r1_acc - base_acc)
        single_random = np.array(single_random)

        all_results[phase_name] = {
            "step": step,
            "base_acc": float(base_acc),
            "edge_delta": float(edge_delta),
            "random_mean": float(random_deltas.mean()),
            "random_std": float(random_deltas.std()),
            "random_all": random_deltas.tolist(),
            "v1_delta": float(v1_acc - base_acc),
            "single_random_mean": float(single_random.mean()),
            "single_random_std": float(single_random.std()),
            "single_random_all": single_random.tolist(),
        }

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ablation: Edge vs Random Directions (Grokked Dyck)", fontsize=14)

    # 2-direction ablation
    ax = axes[0]
    edge_vals = [all_results[p]["edge_delta"] for p in phases]
    rand_means = [all_results[p]["random_mean"] for p in phases]
    rand_stds = [all_results[p]["random_std"] for p in phases]

    x = np.arange(len(phases))
    ax.bar(x - 0.15, edge_vals, 0.3, color="steelblue", label="Remove edge (v1+v2)")
    ax.bar(x + 0.15, rand_means, 0.3, color="lightcoral",
           yerr=rand_stds, capsize=4, label=f"Remove random 2-dir (n={N_RANDOM})")
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=8)
    ax.set_ylabel("Δ accuracy")
    ax.set_title("2-direction ablation")
    ax.legend(fontsize=8)
    ax.axhline(y=0, ls='-', color='gray', alpha=0.3)

    # 1-direction ablation
    ax = axes[1]
    v1_vals = [all_results[p]["v1_delta"] for p in phases]
    sr_means = [all_results[p]["single_random_mean"] for p in phases]
    sr_stds = [all_results[p]["single_random_std"] for p in phases]

    ax.bar(x - 0.15, v1_vals, 0.3, color="steelblue", label="Remove v1 only")
    ax.bar(x + 0.15, sr_means, 0.3, color="lightcoral",
           yerr=sr_stds, capsize=4, label=f"Remove random 1-dir (n={N_RANDOM})")
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=8)
    ax.set_ylabel("Δ accuracy")
    ax.set_title("1-direction ablation")
    ax.legend(fontsize=8)
    ax.axhline(y=0, ls='-', color='gray', alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "random_direction_controls.png", dpi=150)
    plt.close(fig)

    # Save
    torch.save(all_results, FIG_DIR / "random_direction_controls_results.pt")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Random Direction Controls")
    print("="*70)
    for p in phases:
        r = all_results[p]
        print(f"  {p} (step {r['step']}):")
        print(f"    Edge (v1+v2): Δacc = {r['edge_delta']:+.3f}")
        print(f"    Random 2-dir: Δacc = {r['random_mean']:+.3f} ± {r['random_std']:.3f}")
        print(f"    v1 only:      Δacc = {r['v1_delta']:+.3f}")
        print(f"    Random 1-dir: Δacc = {r['single_random_mean']:+.3f} ± {r['single_random_std']:.3f}")
        ratio = abs(r['edge_delta']) / (abs(r['random_mean']) + 1e-8)
        print(f"    Edge is {ratio:.1f}x more impactful than random")


if __name__ == "__main__":
    main()
