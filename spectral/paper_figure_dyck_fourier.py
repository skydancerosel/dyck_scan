#!/usr/bin/env python3
"""
Paper-quality multi-panel figure for Dyck Fourier functional analysis.

5-panel figure:
  A: Power spectrum pre/post grokking (from fourier_functional_dyck.py)
  B: Depth-conditioned representation PCA (from dyck_depth_basis.py)
  C: Layer-wise probe R² curves (from dyck_intermediate_probing.py)
  D: Attention head mode spectra (from dyck_attention_modes.py)
  E: Composition test cross-term R² (from dyck_composition_test.py)
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

from dyck.grok_sweep import DyckTransformerLM, VOCAB_SIZE, build_depth_dataset, split_dataset
from spectral.fourier_functional_dyck import (
    load_model_at_step, extract_hidden_reps, compute_positional_power_spectrum,
    compute_spectral_concentration
)
from spectral.dyck_depth_basis import depth_conditioned_analysis
from spectral.dyck_intermediate_probing import probe_layer
from spectral.dyck_composition_test import composition_probe, build_composition_features
from spectral.dyck_attention_modes import extract_attention_patterns, analyze_attention_spectra

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # Generate test data
    X_all, Y_all = build_depth_dataset(n_seqs=5050, max_pairs=12, ctx_len=24, seed=0)
    _, _, test_x, test_y = split_dataset(X_all, Y_all, frac_train=50/5050, seed=0)
    test_x = test_x[:300]
    test_y = test_y[:300]
    mask = (test_y != -100).numpy()
    depths = test_y.numpy()
    tokens = test_x.numpy()

    # Load models at key steps
    print("Loading models...")
    grok_early_model, _, grok_early_step = load_model_at_step(
        CKPT_DIR / "dyck_grok_fourier.pt", 200)
    grok_late_model, _, grok_late_step = load_model_at_step(
        CKPT_DIR / "dyck_grok_fourier.pt", 5000)
    memo_late_model, _, memo_late_step = load_model_at_step(
        CKPT_DIR / "dyck_memo_fourier.pt", 5000)

    # ══════════════════════════════════════════════════════════════════
    # Compute all panel data
    # ══════════════════════════════════════════════════════════════════

    # Panel A: Power spectra
    print("Computing power spectra...")
    reps_grok_early = extract_hidden_reps(grok_early_model, test_x)
    reps_grok_late = extract_hidden_reps(grok_late_model, test_x)
    reps_memo_late = extract_hidden_reps(memo_late_model, test_x)

    freqs_ge, power_ge = compute_positional_power_spectrum(reps_grok_early["layer_1"], mask)
    freqs_gl, power_gl = compute_positional_power_spectrum(reps_grok_late["layer_1"], mask)
    freqs_ml, power_ml = compute_positional_power_spectrum(reps_memo_late["layer_1"], mask)

    # Panel B: Depth centroids PCA
    print("Computing depth geometry...")
    depth_grok = depth_conditioned_analysis(reps_grok_late["layer_1"], depths, mask)
    depth_memo = depth_conditioned_analysis(reps_memo_late["layer_1"], depths, mask)

    # Panel C: Probing R² at each layer
    print("Computing probes...")
    probe_results = {}
    for name, reps in [("grok", reps_grok_late), ("memo", reps_memo_late)]:
        probe_results[name] = {}
        for layer_name in ["embedding", "layer_0", "layer_1"]:
            probe_results[name][layer_name] = probe_layer(reps[layer_name], depths, mask)

    # Panel D: Attention patterns
    print("Computing attention patterns...")
    attn_grok = extract_attention_patterns(grok_late_model, test_x)
    attn_memo = extract_attention_patterns(memo_late_model, test_x)
    attn_analysis_grok = analyze_attention_spectra(attn_grok, mask)
    attn_analysis_memo = analyze_attention_spectra(attn_memo, mask)

    # Panel E: Composition test
    print("Computing composition...")
    comp_grok = composition_probe(reps_grok_late["layer_1"], tokens, depths, mask)
    comp_memo = composition_probe(reps_memo_late["layer_1"], tokens, depths, mask)

    # ══════════════════════════════════════════════════════════════════
    # Create figure
    # ══════════════════════════════════════════════════════════════════

    print("Generating figure...")
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ── Panel A: Power spectrum (top-left, spans 2 cols) ──
    ax_a = fig.add_subplot(gs[0, 0:2])
    omega = np.arange(len(power_gl))
    width = 0.25
    ax_a.bar(omega - width, power_ge / power_ge.max(), width,
             color='lightgray', label=f'Grok pre (step {grok_early_step})', alpha=0.8)
    ax_a.bar(omega, power_gl / power_gl.max(), width,
             color='steelblue', label=f'Grok post (step {grok_late_step})', alpha=0.8)
    ax_a.bar(omega + width, power_ml / power_ml.max(), width,
             color='coral', label=f'Memo (step {memo_late_step})', alpha=0.8)
    ax_a.set_xlabel("Frequency ω (cycles/position)")
    ax_a.set_ylabel("Normalized |H(ω)|²")
    ax_a.set_title("A. Positional Fourier Power Spectrum (Layer 1)", fontsize=12, fontweight='bold')
    ax_a.legend(fontsize=8)

    conc_gl = compute_spectral_concentration(power_gl)
    conc_ml = compute_spectral_concentration(power_ml)
    ax_a.text(0.98, 0.95, f"Grok: top3={conc_gl['top3']:.2f}, ω*={conc_gl['dominant_omega']}\n"
                           f"Memo: top3={conc_ml['top3']:.2f}, ω*={conc_ml['dominant_omega']}",
              transform=ax_a.transAxes, ha='right', va='top', fontsize=8,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ── Panel B: Depth centroids PCA (top-right) ──
    ax_b = fig.add_subplot(gs[0, 2])
    for data, tag, color in [(depth_grok, "Grok", "steelblue"), (depth_memo, "Memo", "coral")]:
        pca_coords = data["centroid_pca"]
        dk = data["depth_keys"]
        ax_b.plot(pca_coords[:, 0], pca_coords[:, 1], 'o-', color=color,
                  label=f"{tag} (R²={data['linear_r2']:.2f})", markersize=5, alpha=0.7)
        # Label first and last
        ax_b.annotate(f"d={dk[0]}", (pca_coords[0, 0], pca_coords[0, 1]),
                     fontsize=7, color=color)
        ax_b.annotate(f"d={dk[-1]}", (pca_coords[-1, 0], pca_coords[-1, 1]),
                     fontsize=7, color=color)
    ax_b.set_xlabel(f"PC1 ({depth_grok['explained_var'][0]:.1%})")
    ax_b.set_ylabel(f"PC2 ({depth_grok['explained_var'][1]:.1%})")
    ax_b.set_title("B. Depth Centroid Geometry (Layer 1)", fontsize=12, fontweight='bold')
    ax_b.legend(fontsize=8)

    # ── Panel C: Probing R² (middle-left) ──
    ax_c = fig.add_subplot(gs[1, 0])
    layer_names = ["embedding", "layer_0", "layer_1"]
    layer_labels = ["Embedding", "Layer 0", "Layer 1"]
    x_pos = np.arange(len(layer_names))
    width = 0.3
    grok_r2 = [probe_results["grok"][ln]["r2"] for ln in layer_names]
    memo_r2 = [probe_results["memo"][ln]["r2"] for ln in layer_names]
    ax_c.bar(x_pos - width/2, grok_r2, width, color='steelblue', label='Grok')
    ax_c.bar(x_pos + width/2, memo_r2, width, color='coral', label='Memo')
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(layer_labels)
    ax_c.set_ylabel("R² (depth probe)")
    ax_c.set_title("C. Layer-wise Depth Probing", fontsize=12, fontweight='bold')
    ax_c.legend(fontsize=8)
    ax_c.set_ylim(0, 1.1)

    # ── Panel D: Attention patterns (middle, spans 2 cols) ──
    # Show 4 key heads for grok model
    for h_idx in range(4):
        ax_d = fig.add_subplot(gs[1, 1]) if h_idx == 0 else fig.add_subplot(gs[1, 2]) if h_idx == 1 else None
        if ax_d is None:
            continue
        layer = 1 if h_idx < 2 else 0
        head = h_idx % 2
        r = attn_analysis_grok[(layer, head)]
        im = ax_d.imshow(attn_grok[(layer, head)].mean(axis=0), aspect='auto',
                        cmap='viridis', vmin=0)
        ax_d.set_xlabel("key position")
        ax_d.set_ylabel("query position")
        ent = r['mean_entropy']
        kl = r['counting_kl']
        title_prefix = "D" if h_idx == 0 else ""
        ax_d.set_title(f"{title_prefix}. Grok L{layer}H{head} (ent={ent:.1f}, KL={kl:.2f})",
                       fontsize=10, fontweight='bold' if h_idx == 0 else 'normal')

    # ── Panel E: Composition test (bottom row, spans full width) ──
    ax_e = fig.add_subplot(gs[2, :])
    keys = [
        "token→depth", "position→depth", "cumsum→depth",
        "token+pos→depth", "token+pos+cross→depth",
        "full_rep→depth", "rep→cumsum", "rep→token_sign"
    ]
    labels = [
        "token\nonly", "position\nonly", "cumsum\n(oracle)",
        "tok+pos", "tok+pos\n+cross", "full\nrep→depth",
        "rep→\ncumsum", "rep→\ntok_sign"
    ]
    x_pos = np.arange(len(keys))
    width = 0.35
    grok_vals = [comp_grok.get(k, 0) for k in keys]
    memo_vals = [comp_memo.get(k, 0) for k in keys]
    ax_e.bar(x_pos - width/2, grok_vals, width, color='steelblue', label='Grok')
    ax_e.bar(x_pos + width/2, memo_vals, width, color='coral', label='Memo')
    ax_e.set_xticks(x_pos)
    ax_e.set_xticklabels(labels, fontsize=8)
    ax_e.set_ylabel("R²")
    ax_e.set_title("E. Compositional Structure: Token × Accumulation Factorization",
                   fontsize=12, fontweight='bold')
    ax_e.legend(fontsize=9)
    ax_e.set_ylim(-0.1, 1.15)
    ax_e.axhline(y=0, ls='-', color='gray', alpha=0.3)

    # Cross-term boost annotation
    grok_boost = comp_grok.get("token+pos+cross→depth", 0) - comp_grok.get("token+pos→depth", 0)
    memo_boost = comp_memo.get("token+pos+cross→depth", 0) - comp_memo.get("token+pos→depth", 0)
    ax_e.annotate(f"Cross-term boost:\nGrok: +{grok_boost:.3f}\nMemo: +{memo_boost:.3f}",
                  xy=(4, max(grok_vals[4], memo_vals[4])), xytext=(5.5, 0.85),
                  fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'),
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle("Fourier Functional Modes of Dyck-1 Depth Prediction",
                fontsize=16, fontweight='bold', y=0.98)

    fig.savefig(FIG_DIR / "paper_figure_dyck_fourier.png", dpi=200, bbox_inches='tight')
    fig.savefig(FIG_DIR / "paper_figure_dyck_fourier.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {FIG_DIR / 'paper_figure_dyck_fourier.png'}")
    print(f"Saved: {FIG_DIR / 'paper_figure_dyck_fourier.pdf'}")


if __name__ == "__main__":
    main()
