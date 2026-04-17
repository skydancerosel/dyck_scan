#!/usr/bin/env python3
"""
Paper-quality multi-panel figure for SCAN Fourier functional analysis.

5-panel figure:
  A: Power spectrum encoder/decoder (grok vs memo)
  B: Probing R² across layers
  C: Cross-attention entropy evolution
  D: Composition R² comparison
  E: Spectral concentration over training
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from spectral.fourier_functional_scan import (
    load_scan_model_at_step, extract_scan_hidden_reps,
    compute_positional_power_spectrum, compute_spectral_concentration,
)
from spectral.scan_intermediate_probing import probe_layer_for_target
from spectral.scan_composition_test import extract_command_features, composition_probe_scan
from spectral.scan_attention_modes import extract_scan_attention_patterns, analyze_attention

CKPT_DIR = Path(__file__).resolve().parent / "fourier_scan_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_scan_plots"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # Check checkpoints exist
    grok_path = CKPT_DIR / "scan_grok_fourier.pt"
    memo_path = CKPT_DIR / "scan_memo_fourier.pt"
    if not grok_path.exists() or not memo_path.exists():
        print("Checkpoints not found. Run retrain_scan_fourier.py first.")
        return

    # Load test data
    ckpt_g = torch.load(grok_path, weights_only=False)
    test_src = ckpt_g["test_src"][:150]
    test_tgt_in = ckpt_g["test_tgt_in"][:150]
    test_tgt_out = ckpt_g["test_tgt_out"][:150]
    cmd_vocab = ckpt_g["cmd_vocab"]

    src_mask = (test_src != 0).numpy()
    tgt_mask = (test_tgt_out != -100).numpy()
    tgt_tokens = test_tgt_out.numpy()

    # Load models
    print("Loading models...")
    grok_model, _, grok_step = load_scan_model_at_step(grok_path, 10000)
    memo_model, _, memo_step = load_scan_model_at_step(memo_path, 10000)

    # ── Compute data for all panels ──
    print("Computing representations...")
    reps_grok = extract_scan_hidden_reps(grok_model, test_src, test_tgt_in)
    reps_memo = extract_scan_hidden_reps(memo_model, test_src, test_tgt_in)

    print("Computing attention patterns...")
    attn_grok = extract_scan_attention_patterns(grok_model, test_src[:80], test_tgt_in[:80])
    attn_memo = extract_scan_attention_patterns(memo_model, test_src[:80], test_tgt_in[:80])
    attn_analysis_grok = analyze_attention(attn_grok)
    attn_analysis_memo = analyze_attention(attn_memo)

    print("Computing composition...")
    cmd_feats = extract_command_features(cmd_vocab, test_src)
    enc_reps_g = reps_grok.get("enc_2", reps_grok.get("enc_1"))
    dec_reps_g = reps_grok.get("dec_2", reps_grok.get("dec_1"))
    enc_reps_m = reps_memo.get("enc_2", reps_memo.get("enc_1"))
    dec_reps_m = reps_memo.get("dec_2", reps_memo.get("dec_1"))
    comp_grok = composition_probe_scan(enc_reps_g, dec_reps_g, cmd_feats, tgt_tokens, src_mask, tgt_mask)
    comp_memo = composition_probe_scan(enc_reps_m, dec_reps_m, cmd_feats, tgt_tokens, src_mask, tgt_mask)

    # ══════════════════════════════════════════════════════════════════
    # Create figure
    # ══════════════════════════════════════════════════════════════════
    print("Generating figure...")
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel A: Encoder power spectrum ──
    ax_a = fig.add_subplot(gs[0, 0])
    for layer_name, label, color in [("enc_2", "Grok enc_2", "steelblue"), ("enc_2", "Memo enc_2", "coral")]:
        reps = reps_grok if "Grok" in label else reps_memo
        if layer_name in reps:
            _, power = compute_positional_power_spectrum(reps[layer_name], src_mask)
            conc = compute_spectral_concentration(power)
            omega = np.arange(len(power))
            ax_a.bar(omega + (0 if "Grok" in label else 0.3), power / power.max(), 0.3,
                    color=color, alpha=0.8, label=f"{label} (top3={conc['top3']:.2f})")
    ax_a.set_xlabel("Frequency ω")
    ax_a.set_ylabel("Normalized |H(ω)|²")
    ax_a.set_title("A. Encoder Power Spectrum", fontsize=12, fontweight='bold')
    ax_a.legend(fontsize=7)

    # ── Panel B: Decoder power spectrum ──
    ax_b = fig.add_subplot(gs[0, 1])
    for layer_name, label, color in [("dec_2", "Grok dec_2", "steelblue"), ("dec_2", "Memo dec_2", "coral")]:
        reps = reps_grok if "Grok" in label else reps_memo
        if layer_name in reps:
            _, power = compute_positional_power_spectrum(reps[layer_name], tgt_mask)
            conc = compute_spectral_concentration(power)
            omega = np.arange(len(power))
            ax_b.bar(omega + (0 if "Grok" in label else 0.3), power / power.max(), 0.3,
                    color=color, alpha=0.8, label=f"{label} (top3={conc['top3']:.2f})")
    ax_b.set_xlabel("Frequency ω")
    ax_b.set_ylabel("Normalized |H(ω)|²")
    ax_b.set_title("B. Decoder Power Spectrum", fontsize=12, fontweight='bold')
    ax_b.legend(fontsize=7)

    # ── Panel C: Layer-wise probing ──
    ax_c = fig.add_subplot(gs[0, 2])
    # Probe each layer
    first_action = np.zeros_like(test_src.numpy())
    for i in range(len(test_tgt_out)):
        valid = test_tgt_out[i][test_tgt_out[i] != -100]
        if len(valid) > 0:
            first_action[i, :] = valid[0].item()

    enc_layers = ["src_embedding", "enc_0", "enc_1", "enc_2"]
    for tag, reps, color in [("grok", reps_grok, "steelblue"), ("memo", reps_memo, "coral")]:
        r2_vals = []
        for ln in enc_layers:
            if ln in reps:
                r = probe_layer_for_target(reps[ln], first_action, src_mask)
                r2_vals.append(r["r2"])
            else:
                r2_vals.append(0)
        x_pos = np.arange(len(enc_layers))
        ax_c.bar(x_pos + (0 if tag == "grok" else 0.35), r2_vals, 0.35,
                color=color, label=tag)
    ax_c.set_xticks(np.arange(len(enc_layers)) + 0.175)
    ax_c.set_xticklabels(["emb", "enc0", "enc1", "enc2"], fontsize=8)
    ax_c.set_ylabel("R²")
    ax_c.set_title("C. Encoder → Action Probe", fontsize=12, fontweight='bold')
    ax_c.legend(fontsize=8)

    # ── Panel D: Cross-attention entropy ──
    ax_d = fig.add_subplot(gs[1, 0:2])
    for tag, analysis, color in [("grok", attn_analysis_grok, "steelblue"),
                                   ("memo", attn_analysis_memo, "coral")]:
        for layer in range(3):
            for head in range(4):
                key = ("cross", layer, head)
                if key in analysis:
                    ent = analysis[key]["mean_entropy"]
                    x = layer * 4 + head
                    ax_d.bar(x + (0 if tag == "grok" else 0.4), ent, 0.4,
                            color=color, alpha=0.7)
    ax_d.set_xticks([i for i in range(12)])
    ax_d.set_xticklabels([f"L{l}H{h}" for l in range(3) for h in range(4)], fontsize=7, rotation=45)
    ax_d.set_ylabel("Entropy")
    ax_d.set_title("D. Cross-Attention Entropy Per Head", fontsize=12, fontweight='bold')
    # Legend
    from matplotlib.patches import Patch
    ax_d.legend(handles=[Patch(color="steelblue", label="Grok"), Patch(color="coral", label="Memo")],
                fontsize=8)

    # ── Panel E: Composition R² ──
    ax_e = fig.add_subplot(gs[1, 2])
    keys = sorted(set(list(comp_grok.keys()) + list(comp_memo.keys())))
    short_labels = [k.replace("→", "\n→").replace("enc", "E").replace("dec", "D").replace("cmd_feats", "cmd") for k in keys]
    x_pos = np.arange(len(keys))
    grok_vals = [comp_grok.get(k, 0) for k in keys]
    memo_vals = [comp_memo.get(k, 0) for k in keys]
    ax_e.barh(x_pos - 0.2, grok_vals, 0.35, color='steelblue', label='Grok')
    ax_e.barh(x_pos + 0.2, memo_vals, 0.35, color='coral', label='Memo')
    ax_e.set_yticks(x_pos)
    ax_e.set_yticklabels(short_labels, fontsize=7)
    ax_e.set_xlabel("R²")
    ax_e.set_title("E. Composition Probes", fontsize=12, fontweight='bold')
    ax_e.legend(fontsize=8)
    ax_e.set_xlim(-0.1, 1.1)

    # ── Panel F: Summary table ──
    ax_f = fig.add_subplot(gs[2, :])
    ax_f.axis('off')

    # Build summary text
    summary_lines = []
    for tag, reps, attn_analysis in [
        ("Grokked", reps_grok, attn_analysis_grok),
        ("Memorized", reps_memo, attn_analysis_memo),
    ]:
        enc_conc = compute_spectral_concentration(
            compute_positional_power_spectrum(reps.get("enc_2", reps["enc_0"]), src_mask)[1])
        dec_conc = compute_spectral_concentration(
            compute_positional_power_spectrum(reps.get("dec_2", reps["dec_0"]), tgt_mask)[1])
        cross_ents = [attn_analysis[k]["mean_entropy"] for k in attn_analysis if k[0] == "cross"]
        summary_lines.append(
            f"{tag}: Enc ω*={enc_conc['dominant_omega']} top3={enc_conc['top3']:.2f} | "
            f"Dec ω*={dec_conc['dominant_omega']} top3={dec_conc['top3']:.2f} | "
            f"Cross-attn entropy={np.mean(cross_ents):.2f}"
        )

    summary_text = "\n".join(summary_lines)
    ax_f.text(0.5, 0.5, summary_text, transform=ax_f.transAxes,
              ha='center', va='center', fontsize=11, family='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle("Fourier Functional Modes of SCAN Command-to-Action Translation",
                fontsize=16, fontweight='bold', y=0.98)

    fig.savefig(FIG_DIR / "paper_figure_scan_fourier.png", dpi=200, bbox_inches='tight')
    fig.savefig(FIG_DIR / "paper_figure_scan_fourier.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {FIG_DIR / 'paper_figure_scan_fourier.png'}")
    print(f"Saved: {FIG_DIR / 'paper_figure_scan_fourier.pdf'}")


if __name__ == "__main__":
    main()
