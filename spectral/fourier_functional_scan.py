#!/usr/bin/env python3
"""
Fourier functional analysis of SCAN command-to-action translation.

Adapts the Dyck positional DFT analysis for encoder-decoder architecture:
- Encoder: DFT of hidden reps along command positions
- Decoder: DFT of hidden reps along action positions
- Compare grokked vs memorized models
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

from scan.grok_sweep import ScanTransformer

CKPT_DIR = Path(__file__).resolve().parent / "fourier_scan_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_scan_plots"


def load_scan_model_at_step(ckpt_path, step_target):
    """Load SCAN checkpoint and reconstruct model at a specific training step."""
    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg = ckpt["cfg"]

    model = ScanTransformer(
        src_vocab_size=ckpt["cmd_vocab"].size,
        tgt_vocab_size=ckpt["act_vocab"].size,
        max_src_len=ckpt["max_cmd_len"],
        max_tgt_len=ckpt["max_act_len"],
        d_model=cfg["D_MODEL"], n_layers=cfg["N_LAYERS"],
        n_heads=cfg["N_HEADS"], d_ff=cfg["D_FF"],
        dropout=cfg["DROPOUT"],
    )

    best_snap = min(ckpt["snapshots"], key=lambda s: abs(s["step"] - step_target))
    model.load_state_dict(best_snap["state_dict"])
    model.eval()
    return model, ckpt, best_snap["step"]


def extract_scan_hidden_reps(model, src, tgt_in, device="cpu"):
    """Extract hidden representations from encoder and decoder layers.

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

    # Hook encoder layers
    for i, layer in enumerate(model.transformer.encoder.layers):
        hooks.append(layer.register_forward_hook(make_hook(f"enc_{i}")))

    # Hook decoder layers
    for i, layer in enumerate(model.transformer.decoder.layers):
        hooks.append(layer.register_forward_hook(make_hook(f"dec_{i}")))

    with torch.no_grad():
        src_dev = src.to(device)
        tgt_dev = tgt_in.to(device)
        B, S = src_dev.shape
        _, T = tgt_dev.shape

        # Compute embeddings manually
        src_pos = torch.arange(S, device=device)
        src_emb = model.src_tok_emb(src_dev) + model.src_pos_emb(src_pos)[None, :, :]
        reps["src_embedding"] = src_emb.detach().cpu()

        tgt_pos = torch.arange(T, device=device)
        tgt_emb = model.tgt_tok_emb(tgt_dev) + model.tgt_pos_emb(tgt_pos)[None, :, :]
        reps["tgt_embedding"] = tgt_emb.detach().cpu()

        # Full forward pass
        pad_id = 0
        src_pad_mask = (src_dev == pad_id)
        tgt_pad_mask = (tgt_dev == pad_id)
        logits = model(src_dev, tgt_dev, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)

    for h in hooks:
        h.remove()

    return {k: v.numpy() for k, v in reps.items()}


def compute_positional_power_spectrum(reps, mask=None):
    """Compute DFT power spectrum along position dimension."""
    N, T, D = reps.shape
    if mask is not None:
        reps = reps * mask[:, :, None]
    F = np.fft.rfft(reps, axis=1)
    power = np.mean(np.abs(F) ** 2, axis=(0, 2))
    freqs = np.fft.rfftfreq(T, d=1.0)
    return freqs, power


def compute_spectral_concentration(power):
    """Compute spectral concentration ratio."""
    total = power.sum()
    sorted_power = np.sort(power)[::-1]
    top1 = sorted_power[0] / total
    top3 = sorted_power[:3].sum() / total
    top5 = sorted_power[:5].sum() / total
    dominant_freq = np.argmax(power)
    return {"top1": top1, "top3": top3, "top5": top5, "dominant_omega": dominant_freq}


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    analysis_steps = [0, 1000, 2000, 3000, 5000, 10000]

    results = {}

    for tag, ckpt_name in [("grok", "scan_grok_fourier.pt"), ("memo", "scan_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"Skipping {tag}: {ckpt_path} not found")
            continue

        print(f"\n{'='*50}")
        print(f"Analyzing {tag} model")
        print(f"{'='*50}")

        ckpt = torch.load(ckpt_path, weights_only=False)
        test_src = ckpt["test_src"][:200]
        test_tgt_in = ckpt["test_tgt_in"][:200]
        test_tgt_out = ckpt["test_tgt_out"][:200]

        src_mask = (test_src != 0).numpy()  # valid command positions
        tgt_mask = (test_tgt_out != -100).numpy()  # valid action positions

        results[tag] = {}

        for target_step in analysis_steps:
            model, _, actual_step = load_scan_model_at_step(ckpt_path, target_step)
            print(f"\n  Step {actual_step}:")

            reps = extract_scan_hidden_reps(model, test_src, test_tgt_in)

            step_results = {}

            # Encoder layers
            for layer_name in ["src_embedding", "enc_0", "enc_1", "enc_2"]:
                if layer_name not in reps:
                    continue
                freqs, power = compute_positional_power_spectrum(reps[layer_name], src_mask)
                conc = compute_spectral_concentration(power)
                step_results[layer_name] = {"freqs": freqs, "power": power, "concentration": conc}
                print(f"    {layer_name}: ω*={conc['dominant_omega']}, top1={conc['top1']:.3f}, top3={conc['top3']:.3f}")

            # Decoder layers
            for layer_name in ["tgt_embedding", "dec_0", "dec_1", "dec_2"]:
                if layer_name not in reps:
                    continue
                freqs, power = compute_positional_power_spectrum(reps[layer_name], tgt_mask)
                conc = compute_spectral_concentration(power)
                step_results[layer_name] = {"freqs": freqs, "power": power, "concentration": conc}
                print(f"    {layer_name}: ω*={conc['dominant_omega']}, top1={conc['top1']:.3f}, top3={conc['top3']:.3f}")

            results[tag][actual_step] = step_results

    if not results:
        print("No checkpoints found. Run retrain_scan_fourier.py first.")
        return

    # ── Plot 1: Encoder power spectra (grok vs memo) ──
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Encoder Positional Fourier Power Spectrum: Grokked vs Memorized", fontsize=14)

    for row, tag in enumerate(["grok", "memo"]):
        if tag not in results:
            continue
        steps = sorted(results[tag].keys())
        late = steps[-1]
        for col, layer_name in enumerate(["src_embedding", "enc_0", "enc_1", "enc_2"]):
            ax = axes[row, col]
            if layer_name in results[tag][late]:
                data = results[tag][late][layer_name]
                omega = np.arange(len(data["power"]))
                ax.bar(omega, data["power"], color="steelblue" if tag == "grok" else "coral", alpha=0.8)
                conc = data["concentration"]
                ax.set_title(f"{tag} step={late}: {layer_name}\ntop1={conc['top1']:.2f}, ω*={conc['dominant_omega']}", fontsize=9)
            ax.set_xlabel("ω")
            ax.set_ylabel("|H(ω)|²")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "encoder_power_spectrum.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Decoder power spectra ──
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Decoder Positional Fourier Power Spectrum: Grokked vs Memorized", fontsize=14)

    for row, tag in enumerate(["grok", "memo"]):
        if tag not in results:
            continue
        steps = sorted(results[tag].keys())
        late = steps[-1]
        for col, layer_name in enumerate(["tgt_embedding", "dec_0", "dec_1", "dec_2"]):
            ax = axes[row, col]
            if layer_name in results[tag][late]:
                data = results[tag][late][layer_name]
                omega = np.arange(len(data["power"]))
                ax.bar(omega, data["power"], color="steelblue" if tag == "grok" else "coral", alpha=0.8)
                conc = data["concentration"]
                ax.set_title(f"{tag} step={late}: {layer_name}\ntop1={conc['top1']:.2f}, ω*={conc['dominant_omega']}", fontsize=9)
            ax.set_xlabel("ω")
            ax.set_ylabel("|H(ω)|²")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "decoder_power_spectrum.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Spectral concentration over training ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Spectral Concentration During Training (top-3 energy fraction)", fontsize=14)

    for col, (side, layers) in enumerate([
        ("Encoder", ["src_embedding", "enc_0", "enc_1", "enc_2"]),
        ("Decoder", ["tgt_embedding", "dec_0", "dec_1", "dec_2"]),
    ]):
        ax = axes[col]
        for tag, ls in [("grok", "-"), ("memo", "--")]:
            if tag not in results:
                continue
            steps_sorted = sorted(results[tag].keys())
            # Use deepest layer
            deep_layer = layers[-1]
            top3_vals = []
            for s in steps_sorted:
                if deep_layer in results[tag][s]:
                    top3_vals.append(results[tag][s][deep_layer]["concentration"]["top3"])
                else:
                    top3_vals.append(np.nan)
            color = "steelblue" if tag == "grok" else "coral"
            ax.plot(steps_sorted, top3_vals, f'o{ls}', color=color, label=f"{tag} ({deep_layer})", markersize=4)
        ax.set_title(f"{side} (deepest layer)")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Top-3 spectral concentration")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "spectral_concentration_training.png", dpi=150)
    plt.close(fig)

    # Save numerical results
    summary = {}
    for tag in results:
        summary[tag] = {}
        for step in sorted(results[tag].keys()):
            summary[tag][step] = {}
            for layer_name in results[tag][step]:
                summary[tag][step][layer_name] = results[tag][step][layer_name]["concentration"]
    torch.save(summary, FIG_DIR / "fourier_results.pt")

    print(f"\nSaved figures to {FIG_DIR}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: SCAN Positional Fourier Analysis")
    print("="*60)
    for tag in results:
        steps = sorted(results[tag].keys())
        late = steps[-1]
        print(f"\n{tag} (step {late}):")
        for ln in sorted(results[tag][late].keys()):
            c = results[tag][late][ln]["concentration"]
            print(f"  {ln}: ω*={c['dominant_omega']}, top1={c['top1']:.3f}, top3={c['top3']:.3f}")


if __name__ == "__main__":
    main()
