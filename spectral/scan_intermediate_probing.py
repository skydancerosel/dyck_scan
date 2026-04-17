#!/usr/bin/env python3
"""
Layer-wise intermediate probing for SCAN command-to-action translation.

Fits linear probes at each encoder and decoder layer to predict:
- Encoder: command semantics (action type, repetition count)
- Decoder: next action token, sequence position

Tracks R² across training steps, comparing grokked vs memorized.
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

from spectral.fourier_functional_scan import load_scan_model_at_step, extract_scan_hidden_reps

CKPT_DIR = Path(__file__).resolve().parent / "fourier_scan_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_scan_plots"


def probe_layer_for_target(reps, targets, mask):
    """Fit linear probe. Returns R² and accuracy."""
    N, T, D = reps.shape
    reps_flat = reps.reshape(-1, D)
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1).astype(bool)

    X = reps_flat[mask_flat]
    y = targets_flat[mask_flat]

    if len(np.unique(y)) < 2:
        return {"r2": 0.0, "accuracy": 0.0}

    n = len(X)
    perm = np.random.permutation(n)
    split = int(0.7 * n)
    X_train, X_test = X[perm[:split]], X[perm[split:]]
    y_train, y_test = y[perm[:split]], y[perm[split:]]

    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)
    pred = probe.predict(X_test)
    r2 = r2_score(y_test, pred)

    pred_class = np.clip(np.round(pred), y.min(), y.max()).astype(int)
    acc = accuracy_score(y_test, pred_class)

    return {"r2": r2, "accuracy": acc}


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    analysis_steps = [0, 1000, 2000, 3000, 5000, 10000]

    results = {}

    for tag, ckpt_name in [("grok", "scan_grok_fourier.pt"), ("memo", "scan_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"Skipping {tag}: not found")
            continue

        ckpt = torch.load(ckpt_path, weights_only=False)
        test_src = ckpt["test_src"][:200]
        test_tgt_in = ckpt["test_tgt_in"][:200]
        test_tgt_out = ckpt["test_tgt_out"][:200]

        # Targets for probing
        tgt_mask = (test_tgt_out != -100).numpy()
        tgt_tokens = test_tgt_out.numpy()  # action token IDs

        # Encoder targets: use the first action token as a proxy for command semantics
        # (what action type the command produces)
        src_mask = (test_src != 0).numpy()
        # Broadcast first target token to all encoder positions
        first_action = np.zeros_like(test_src.numpy())
        for i in range(len(test_tgt_out)):
            valid = test_tgt_out[i][test_tgt_out[i] != -100]
            if len(valid) > 0:
                first_action[i, :] = valid[0].item()

        print(f"\n{'='*50}")
        print(f"Probing: {tag}")
        print(f"{'='*50}")

        results[tag] = {}

        enc_layers = ["src_embedding", "enc_0", "enc_1", "enc_2"]
        dec_layers = ["tgt_embedding", "dec_0", "dec_1", "dec_2"]

        for target_step in analysis_steps:
            model, _, actual_step = load_scan_model_at_step(ckpt_path, target_step)
            reps = extract_scan_hidden_reps(model, test_src, test_tgt_in)

            step_results = {}

            # Encoder: probe for first action token (command semantics)
            for layer_name in enc_layers:
                if layer_name in reps:
                    r = probe_layer_for_target(reps[layer_name], first_action, src_mask)
                    step_results[f"{layer_name}→action"] = r

            # Decoder: probe for next action token
            for layer_name in dec_layers:
                if layer_name in reps:
                    r = probe_layer_for_target(reps[layer_name], tgt_tokens, tgt_mask)
                    step_results[f"{layer_name}→token"] = r

            results[tag][actual_step] = step_results

            if actual_step % 2000 == 0 or actual_step == 0:
                print(f"  Step {actual_step}:")
                for k, v in step_results.items():
                    print(f"    {k}: R²={v['r2']:.3f}, acc={v['accuracy']:.3f}")

    if not results:
        print("No checkpoints found.")
        return

    # ── Plot 1: Encoder probing R² ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, tag in enumerate(["grok", "memo"]):
        if tag not in results:
            continue
        ax = axes[i]
        steps_sorted = sorted(results[tag].keys())
        for layer_name, color, marker in [
            ("src_embedding→action", "gray", "s"),
            ("enc_0→action", "lightblue", "o"),
            ("enc_1→action", "steelblue", "^"),
            ("enc_2→action", "darkblue", "D"),
        ]:
            vals = [results[tag][s].get(layer_name, {}).get("r2", np.nan) for s in steps_sorted]
            ax.plot(steps_sorted, vals, f'{marker}-', color=color, label=layer_name.split("→")[0], markersize=4)
        ax.set_title(f"{tag}: Encoder → First Action R²")
        ax.set_xlabel("Training step")
        ax.set_ylabel("R²")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "encoder_probing_r2.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Decoder probing R² ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, tag in enumerate(["grok", "memo"]):
        if tag not in results:
            continue
        ax = axes[i]
        steps_sorted = sorted(results[tag].keys())
        for layer_name, color, marker in [
            ("tgt_embedding→token", "gray", "s"),
            ("dec_0→token", "lightsalmon", "o"),
            ("dec_1→token", "coral", "^"),
            ("dec_2→token", "darkred", "D"),
        ]:
            vals = [results[tag][s].get(layer_name, {}).get("r2", np.nan) for s in steps_sorted]
            ax.plot(steps_sorted, vals, f'{marker}-', color=color, label=layer_name.split("→")[0], markersize=4)
        ax.set_title(f"{tag}: Decoder → Action Token R²")
        ax.set_xlabel("Training step")
        ax.set_ylabel("R²")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "decoder_probing_r2.png", dpi=150)
    plt.close(fig)

    # Save results
    torch.save(results, FIG_DIR / "probing_results.pt")
    print(f"\nSaved figures to {FIG_DIR}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: SCAN Intermediate Probing")
    print("="*60)
    for tag in results:
        steps = sorted(results[tag].keys())
        late = steps[-1]
        print(f"\n{tag} (step {late}):")
        for k, v in sorted(results[tag][late].items()):
            print(f"  {k}: R²={v['r2']:.3f}, acc={v['accuracy']:.3f}")


if __name__ == "__main__":
    main()
