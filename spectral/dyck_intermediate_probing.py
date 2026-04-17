#!/usr/bin/env python3
"""
Layer-wise intermediate probing for Dyck-1 depth prediction.

Analogous to x2y2_intermediate_probing.py — fits linear probes at each
layer to predict depth, tracking R² across training steps.

Key question: Does depth information emerge suddenly at grokking,
or gradually? Where in the network does it emerge first?
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import cross_val_score

from dyck.grok_sweep import DyckTransformerLM, VOCAB_SIZE, build_depth_dataset, split_dataset
from spectral.fourier_functional_dyck import load_model_at_step, extract_hidden_reps

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"


def probe_layer(reps, targets, mask):
    """Fit linear probe for depth prediction.

    Args:
        reps: [N, T, D]
        targets: [N, T] depth labels (-100 for padding)
        mask: [N, T] boolean

    Returns dict with R², accuracy, and probe weights.
    """
    N, T, D = reps.shape
    reps_flat = reps.reshape(-1, D)
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1).astype(bool)

    X = reps_flat[mask_flat]
    y = targets_flat[mask_flat]

    # Split into train/test for honest evaluation
    n = len(X)
    split = int(0.7 * n)
    perm = np.random.permutation(n)
    X_train, X_test = X[perm[:split]], X[perm[split:]]
    y_train, y_test = y[perm[:split]], y[perm[split:]]

    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)

    pred = probe.predict(X_test)
    r2 = r2_score(y_test, pred)

    # Classification accuracy (round to nearest integer depth)
    pred_class = np.clip(np.round(pred), 0, 12).astype(int)
    acc = accuracy_score(y_test, pred_class)

    return {"r2": r2, "accuracy": acc, "probe_norm": np.linalg.norm(probe.coef_)}


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # Generate test data
    X_all, Y_all = build_depth_dataset(n_seqs=5050, max_pairs=12, ctx_len=24, seed=0)
    _, _, test_x, test_y = split_dataset(X_all, Y_all, frac_train=50/5050, seed=0)
    test_x = test_x[:500]
    test_y = test_y[:500]
    mask = (test_y != -100).numpy()
    depths = test_y.numpy()

    # Dense step sampling for grok model to capture transition
    grok_steps = list(range(0, 2001, 100)) + [3000, 5000]
    memo_steps = [0, 500, 1000, 2000, 5000]

    results = {}

    for tag, ckpt_name, steps in [
        ("grok", "dyck_grok_fourier.pt", grok_steps),
        ("memo", "dyck_memo_fourier.pt", memo_steps),
    ]:
        ckpt_path = CKPT_DIR / ckpt_name
        print(f"\n{'='*50}")
        print(f"Probing: {tag}")
        print(f"{'='*50}")

        results[tag] = {}
        layer_names = ["embedding", "layer_0", "layer_1"]

        for target_step in steps:
            model, cfg, actual_step = load_model_at_step(ckpt_path, target_step)
            reps = extract_hidden_reps(model, test_x)

            step_results = {}
            for layer_name in layer_names:
                if layer_name not in reps:
                    continue
                probe_result = probe_layer(reps[layer_name], depths, mask)
                step_results[layer_name] = probe_result

            results[tag][actual_step] = step_results
            if actual_step % 500 == 0 or actual_step <= 200:
                print(f"  Step {actual_step}: " +
                      ", ".join(f"{ln}={step_results[ln]['r2']:.3f}"
                                for ln in layer_names if ln in step_results))

    # ── Plot 1: R² evolution during grokking ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, tag in enumerate(["grok", "memo"]):
        ax = axes[i]
        steps_sorted = sorted(results[tag].keys())
        for layer_name, color, marker in [
            ("embedding", "gray", "s"),
            ("layer_0", "steelblue", "o"),
            ("layer_1", "darkred", "^"),
        ]:
            r2_vals = [results[tag][s][layer_name]["r2"] for s in steps_sorted]
            ax.plot(steps_sorted, r2_vals, f'{marker}-', color=color,
                    label=layer_name, markersize=4)

        ax.set_title(f"{tag}: Linear Probe R² for Depth")
        ax.set_xlabel("Training step")
        ax.set_ylabel("R² (depth prediction)")
        ax.legend()
        ax.set_ylim(-0.1, 1.05)
        if tag == "grok":
            ax.axvspan(500, 1000, alpha=0.1, color='green', label='grokking region')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "probing_r2_evolution.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {FIG_DIR / 'probing_r2_evolution.png'}")

    # ── Plot 2: Classification accuracy evolution ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, tag in enumerate(["grok", "memo"]):
        ax = axes[i]
        steps_sorted = sorted(results[tag].keys())
        for layer_name, color, marker in [
            ("embedding", "gray", "s"),
            ("layer_0", "steelblue", "o"),
            ("layer_1", "darkred", "^"),
        ]:
            acc_vals = [results[tag][s][layer_name]["accuracy"] for s in steps_sorted]
            ax.plot(steps_sorted, acc_vals, f'{marker}-', color=color,
                    label=layer_name, markersize=4)

        ax.set_title(f"{tag}: Linear Probe Accuracy for Depth")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Accuracy (depth classification)")
        ax.legend()
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "probing_accuracy_evolution.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'probing_accuracy_evolution.png'}")

    # ── Plot 3: Probe weight norm (complexity measure) ──
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_title("Probe Weight Norm During Training (Grokked)", fontsize=14)

    steps_sorted = sorted(results["grok"].keys())
    for layer_name, color in [("embedding", "gray"), ("layer_0", "steelblue"), ("layer_1", "darkred")]:
        norms = [results["grok"][s][layer_name]["probe_norm"] for s in steps_sorted]
        ax.plot(steps_sorted, norms, 'o-', color=color, label=layer_name, markersize=3)

    ax.set_xlabel("Training step")
    ax.set_ylabel("||probe weights||")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "probing_weight_norm.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'probing_weight_norm.png'}")

    # Save results
    torch.save(results, FIG_DIR / "probing_results.pt")
    print(f"Saved: {FIG_DIR / 'probing_results.pt'}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Intermediate Probing")
    print("="*60)
    for tag in ["grok", "memo"]:
        steps = sorted(results[tag].keys())
        late = steps[-1]
        early = steps[1] if len(steps) > 1 else steps[0]
        print(f"\n{tag}:")
        print(f"  Early (step {early}):")
        for ln in ["embedding", "layer_0", "layer_1"]:
            r = results[tag][early][ln]
            print(f"    {ln}: R²={r['r2']:.3f}, acc={r['accuracy']:.3f}")
        print(f"  Late (step {late}):")
        for ln in ["embedding", "layer_0", "layer_1"]:
            r = results[tag][late][ln]
            print(f"    {ln}: R²={r['r2']:.3f}, acc={r['accuracy']:.3f}")


if __name__ == "__main__":
    main()
