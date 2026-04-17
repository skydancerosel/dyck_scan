#!/usr/bin/env python3
"""
Fourier functional analysis of Dyck-1 depth prediction.

Analogous to fourier_functional_view.py for modular arithmetic.
Instead of 2D DFT on Z_p x Z_p, we compute 1D DFT of hidden
representations along the position dimension.

Key question: Does the grokked model concentrate spectral energy
at specific frequencies, while the memorized model is flat?
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

from dyck.grok_sweep import DyckTransformerLM, VOCAB_SIZE, build_depth_dataset, split_dataset

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"


def load_model_at_step(ckpt_path, step_target):
    """Load checkpoint and reconstruct model at a specific training step."""
    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg = ckpt["cfg"]

    model = DyckTransformerLM(
        vocab_size=VOCAB_SIZE,
        ctx_len=max(cfg["CTX_LEN"], cfg["CTX_LEN_OOD"]),
        d_model=cfg["D_MODEL"], n_layers=cfg["N_LAYERS"],
        n_heads=cfg["N_HEADS"], d_ff=cfg["D_FF"],
        dropout=cfg["DROPOUT"], n_classes=cfg["N_CLASSES"],
    )

    # Find closest snapshot
    best_snap = min(ckpt["snapshots"], key=lambda s: abs(s["step"] - step_target))
    model.load_state_dict(best_snap["state_dict"])
    model.eval()
    return model, cfg, best_snap["step"]


def extract_hidden_reps(model, X, device="cpu"):
    """Extract hidden representations at each layer using forward hooks.

    Returns dict: layer_name -> [N, T, d_model] numpy array
    """
    model = model.to(device)
    model.eval()

    reps = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            reps[name] = output.detach().cpu()
        return hook_fn

    # Hook embedding output (before encoder)
    # We'll compute it manually
    # Hook each encoder layer output
    for i, layer in enumerate(model.encoder.layers):
        hooks.append(layer.register_forward_hook(make_hook(f"layer_{i}")))

    with torch.no_grad():
        X_dev = X.to(device)
        B, T = X_dev.shape
        # Manually compute embedding
        pos = torch.arange(T, device=device)
        emb = model.tok_emb(X_dev) + model.pos_emb(pos)[None, :, :]
        reps["embedding"] = emb.detach().cpu()

        # Full forward pass (hooks will capture layer outputs)
        logits = model(X_dev)
        reps["logits"] = logits.detach().cpu()

    for h in hooks:
        h.remove()

    return {k: v.numpy() for k, v in reps.items()}


def compute_positional_power_spectrum(reps, mask=None):
    """Compute DFT power spectrum along position dimension.

    Args:
        reps: [N, T, D] hidden representations
        mask: [N, T] boolean mask (True = valid position)

    Returns:
        freqs: [T//2+1] frequency bins
        power: [T//2+1] mean power spectrum
    """
    N, T, D = reps.shape

    if mask is not None:
        # Zero out padded positions
        reps = reps * mask[:, :, None]

    # DFT along position dimension for each sequence and feature
    # Shape: [N, T//2+1, D] (complex)
    F = np.fft.rfft(reps, axis=1)

    # Power spectrum: |F|^2 averaged over sequences and features
    power = np.mean(np.abs(F) ** 2, axis=(0, 2))  # [T//2+1]

    freqs = np.fft.rfftfreq(T, d=1.0)  # [T//2+1], in cycles per position
    return freqs, power


def compute_spectral_concentration(power):
    """Compute spectral concentration ratio: energy in top-k modes / total."""
    total = power.sum()
    sorted_power = np.sort(power)[::-1]
    top1 = sorted_power[0] / total
    top3 = sorted_power[:3].sum() / total
    top5 = sorted_power[:5].sum() / total
    dominant_freq = np.argmax(power)
    return {"top1": top1, "top3": top3, "top5": top5, "dominant_omega": dominant_freq}


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Generate test sequences
    cfg_dummy = {"N_TOTAL": 5050, "MAX_PAIRS": 12, "CTX_LEN": 24,
                 "N_TRAIN": 50, "DATA_SEED": 0}
    X_all, Y_all = build_depth_dataset(
        n_seqs=cfg_dummy["N_TOTAL"], max_pairs=cfg_dummy["MAX_PAIRS"],
        ctx_len=cfg_dummy["CTX_LEN"], seed=cfg_dummy["DATA_SEED"]
    )
    frac = cfg_dummy["N_TRAIN"] / cfg_dummy["N_TOTAL"]
    _, _, test_x, test_y = split_dataset(X_all, Y_all, frac_train=frac, seed=0)

    # Use first 500 test sequences
    test_x = test_x[:500]
    test_y = test_y[:500]
    mask = (test_y != -100).numpy()  # valid positions

    # Steps to analyze
    analysis_steps = [0, 200, 500, 1000, 2000, 5000]

    results = {}

    for tag, ckpt_name in [("grok", "dyck_grok_fourier.pt"), ("memo", "dyck_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        print(f"\n{'='*50}")
        print(f"Analyzing {tag} model")
        print(f"{'='*50}")

        results[tag] = {}

        for target_step in analysis_steps:
            model, cfg, actual_step = load_model_at_step(ckpt_path, target_step)
            print(f"\n  Step {actual_step}:")

            reps = extract_hidden_reps(model, test_x)

            step_results = {}
            for layer_name in ["embedding", "layer_0", "layer_1"]:
                if layer_name not in reps:
                    continue
                freqs, power = compute_positional_power_spectrum(reps[layer_name], mask)
                conc = compute_spectral_concentration(power)
                step_results[layer_name] = {
                    "freqs": freqs, "power": power, "concentration": conc
                }
                print(f"    {layer_name}: dominant_omega={conc['dominant_omega']}, "
                      f"top1={conc['top1']:.3f}, top3={conc['top3']:.3f}")

            results[tag][actual_step] = step_results

    # ── Plot 1: Power spectrum comparison (grok vs memo, post-training) ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Positional Fourier Power Spectrum: Grokked vs Memorized", fontsize=14)

    for row, tag in enumerate(["grok", "memo"]):
        # Use last available step
        steps_available = sorted(results[tag].keys())
        late_step = steps_available[-1]
        for col, layer_name in enumerate(["embedding", "layer_0", "layer_1"]):
            ax = axes[row, col]
            data = results[tag][late_step][layer_name]
            omega = np.arange(len(data["power"]))
            ax.bar(omega, data["power"], color="steelblue" if tag == "grok" else "coral",
                   alpha=0.8)
            conc = data["concentration"]
            ax.set_title(f"{tag} step={late_step}: {layer_name}\n"
                         f"top1={conc['top1']:.2f}, dominant ω={conc['dominant_omega']}")
            ax.set_xlabel("Frequency ω")
            ax.set_ylabel("|H(ω)|²")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "power_spectrum_grok_vs_memo.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {FIG_DIR / 'power_spectrum_grok_vs_memo.png'}")

    # ── Plot 2: Spectral concentration over training ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Spectral Concentration During Training (top-3 energy fraction)", fontsize=14)

    for col, layer_name in enumerate(["embedding", "layer_0", "layer_1"]):
        ax = axes[col]
        for tag, color in [("grok", "steelblue"), ("memo", "coral")]:
            steps_sorted = sorted(results[tag].keys())
            top3_vals = []
            for s in steps_sorted:
                if layer_name in results[tag][s]:
                    top3_vals.append(results[tag][s][layer_name]["concentration"]["top3"])
                else:
                    top3_vals.append(np.nan)
            ax.plot(steps_sorted, top3_vals, 'o-', color=color, label=tag, markersize=4)
        ax.set_title(layer_name)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Top-3 spectral concentration")
        ax.legend()
        ax.set_ylim(0, 1)
        ax.axhline(y=3.0 / (cfg["CTX_LEN"] // 2 + 1), ls='--', color='gray',
                    alpha=0.5, label="uniform baseline")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "spectral_concentration_training.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'spectral_concentration_training.png'}")

    # ── Plot 3: Dominant frequency evolution ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Dominant Frequency ω* During Training", fontsize=14)

    for col, layer_name in enumerate(["embedding", "layer_0", "layer_1"]):
        ax = axes[col]
        for tag, color in [("grok", "steelblue"), ("memo", "coral")]:
            steps_sorted = sorted(results[tag].keys())
            omegas = []
            for s in steps_sorted:
                if layer_name in results[tag][s]:
                    omegas.append(results[tag][s][layer_name]["concentration"]["dominant_omega"])
                else:
                    omegas.append(np.nan)
            ax.plot(steps_sorted, omegas, 'o-', color=color, label=tag, markersize=4)
        ax.set_title(layer_name)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Dominant frequency ω*")
        ax.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "dominant_freq_training.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'dominant_freq_training.png'}")

    # ── Save numerical results ──
    summary = {}
    for tag in ["grok", "memo"]:
        summary[tag] = {}
        for step in sorted(results[tag].keys()):
            summary[tag][step] = {}
            for layer_name in results[tag][step]:
                summary[tag][step][layer_name] = results[tag][step][layer_name]["concentration"]

    torch.save(summary, FIG_DIR / "fourier_results.pt")
    print(f"Saved: {FIG_DIR / 'fourier_results.pt'}")

    # Print final summary
    print("\n" + "="*60)
    print("SUMMARY: Positional Fourier Analysis")
    print("="*60)
    for tag in ["grok", "memo"]:
        steps_sorted = sorted(results[tag].keys())
        late = steps_sorted[-1]
        print(f"\n{tag} (step {late}):")
        for layer_name in ["embedding", "layer_0", "layer_1"]:
            c = results[tag][late][layer_name]["concentration"]
            print(f"  {layer_name}: ω*={c['dominant_omega']}, "
                  f"top1={c['top1']:.3f}, top3={c['top3']:.3f}, top5={c['top5']:.3f}")


if __name__ == "__main__":
    main()
