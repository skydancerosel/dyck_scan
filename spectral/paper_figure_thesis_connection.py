#!/usr/bin/env python3
"""
Paper figure: Spectral edge thesis connection.

Panel A: WD intervention — accuracy, probe R², entropy vs WD strength
Panel B: Grad vs WD fraction on v1 across grokking phases (both tasks)
Panel C: Hessian curvature — edge vs bulk, grok vs memo
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

FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════
    # Load pre-computed results
    # ══════════════════════════════════════════════════════════════════

    # WD intervention results
    wd_results = torch.load(FIG_DIR / "wd_intervention_results.pt", weights_only=False)

    # Grad vs WD decomposition
    grad_wd = torch.load(FIG_DIR / "grad_vs_wd_results.pt", weights_only=False)

    # Ablation + Hessian
    ablation = torch.load(FIG_DIR / "ablation_hessian_results.pt", weights_only=False)

    # ══════════════════════════════════════════════════════════════════
    # Build figure
    # ══════════════════════════════════════════════════════════════════

    fig = plt.figure(figsize=(16, 5.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.38, left=0.06, right=0.97, top=0.88, bottom=0.15)

    # ── Panel A: WD intervention ──────────────────────────────────────
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = ax_a1.twinx()

    # Extract data from both intervention points (use post_grok as primary)
    point = "post_grok"
    wd_vals_ordered = [0.0, 0.5, 1.0, 2.0]
    cond_map = {0.0: "zero_wd", 0.5: "half_wd", 1.0: "same_wd", 2.0: "double_wd"}

    accs, r2s, ents = [], [], []
    for wd in wd_vals_ordered:
        cond = cond_map[wd]
        data = wd_results[point][cond]
        m_final = data["metrics"][-1]
        d_final = data["diagnostics"][-1]
        accs.append(m_final["test_acc"])
        r2s.append(d_final["probe_r2"])
        ents.append(d_final["attn_entropy"])

    x = np.arange(len(wd_vals_ordered))
    w = 0.25

    # Accuracy bars (left axis)
    bars_acc = ax_a1.bar(x - w, accs, w, color="#2166ac", alpha=0.85, label="Test accuracy", zorder=3)
    # R² bars (left axis, same scale)
    bars_r2 = ax_a1.bar(x, r2s, w, color="#b2182b", alpha=0.85, label="Probe R²", zorder=3)
    # Entropy line (right axis)
    line_ent, = ax_a2.plot(x + w/2, ents, 's-', color="#4daf4a", markersize=8, linewidth=2,
                           label="Attn entropy", zorder=4)

    ax_a1.set_xticks(x)
    ax_a1.set_xticklabels([f"wd={v}" for v in wd_vals_ordered], fontsize=9)
    ax_a1.set_ylabel("Accuracy / R²", fontsize=10)
    ax_a1.set_ylim(0.55, 1.05)
    ax_a1.set_xlabel("Weight decay", fontsize=10)

    ax_a2.set_ylabel("Attention entropy", fontsize=10, color="#4daf4a")
    ax_a2.set_ylim(2.05, 2.32)
    ax_a2.tick_params(axis='y', labelcolor="#4daf4a")

    # Combined legend
    handles = [bars_acc, bars_r2, line_ent]
    labels = ["Test accuracy", "Depth probe R²", "Attn entropy"]
    ax_a1.legend(handles, labels, fontsize=7.5, loc="lower left",
                 framealpha=0.9, edgecolor='gray')

    # Annotation arrows
    ax_a1.annotate("", xy=(0, r2s[0]), xytext=(2, r2s[2]),
                   arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5, ls="--"))
    mid_x = 1.0
    mid_y = (r2s[0] + r2s[2]) / 2 + 0.02
    ax_a1.text(mid_x, mid_y, "WD drives\ncompression", ha="center", fontsize=7,
               color="gray", style="italic")

    ax_a1.set_title("A.  Weight Decay Controls Compression", fontsize=11, fontweight="bold", pad=10)

    # ── Panel B: Grad vs WD on v1 ────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    # Dyck phases
    dyck_phases = ["pre_grok", "at_grok", "post_grok", "late"]
    dyck_labels = ["Pre-\ngrok", "At\ngrok", "Post-\ngrok", "Late"]
    dyck_grad = []
    for p in dyck_phases:
        r = grad_wd["dyck"]["grok"].get(p, {})
        d = r.get("directions", {}).get("v1", {})
        dyck_grad.append(d.get("grad_frac", 0.5))
    dyck_wd = [1 - g for g in dyck_grad]

    # SCAN phases
    scan_phases = ["early", "pre_grok", "at_grok", "post_grok"]
    scan_labels = ["Early", "Pre-\ngrok", "At\ngrok", "Post-\ngrok"]
    scan_grad = []
    for p in scan_phases:
        r = grad_wd["scan"].get("grok", {}).get(p, {})
        d = r.get("directions", {}).get("v1", {})
        scan_grad.append(d.get("grad_frac", 0.5))
    scan_wd = [1 - g for g in scan_grad]

    # Stacked bar chart
    n_dyck = len(dyck_phases)
    n_scan = len(scan_phases)
    x_all = np.arange(n_dyck + n_scan + 1)  # +1 for gap
    x_dyck = x_all[:n_dyck]
    x_scan = x_all[n_dyck + 1:]

    bar_w = 0.7

    # Dyck
    ax_b.bar(x_dyck, dyck_grad, bar_w, color="#2166ac", label="Gradient", zorder=3)
    ax_b.bar(x_dyck, dyck_wd, bar_w, bottom=dyck_grad, color="#f4a582", label="Weight decay", zorder=3)

    # SCAN
    ax_b.bar(x_scan, scan_grad, bar_w, color="#2166ac", zorder=3)
    ax_b.bar(x_scan, scan_wd, bar_w, bottom=scan_grad, color="#f4a582", zorder=3)

    # Labels
    all_labels = dyck_labels + [""] + scan_labels
    ax_b.set_xticks(x_all)
    ax_b.set_xticklabels(all_labels, fontsize=7.5)
    ax_b.set_ylabel("Fraction of v₁ update energy", fontsize=10)
    ax_b.set_ylim(0, 1.05)
    ax_b.axhline(y=0.5, ls=":", color="gray", alpha=0.4, lw=1)
    ax_b.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # Task labels
    ax_b.text(np.mean(x_dyck), -0.18, "Dyck", ha="center", fontsize=10, fontweight="bold",
              transform=ax_b.get_xaxis_transform())
    ax_b.text(np.mean(x_scan), -0.18, "SCAN", ha="center", fontsize=10, fontweight="bold",
              transform=ax_b.get_xaxis_transform())

    # Separator
    sep_x = n_dyck + 0.0
    ax_b.axvline(x=sep_x, ls="-", color="lightgray", lw=1, zorder=1)

    # Annotation: "99.8% WD" at SCAN at_grok
    scan_grok_idx = x_scan[2]  # at_grok position
    ax_b.annotate(f"{scan_wd[2]:.0%}\nWD", xy=(scan_grok_idx, 0.5),
                  fontsize=7, ha="center", va="center", fontweight="bold", color="white",
                  bbox=dict(boxstyle="round,pad=0.2", fc="#d6604d", alpha=0.8))

    ax_b.set_title("B.  Edge Transitions: Gradient → Weight Decay", fontsize=11, fontweight="bold", pad=10)

    # ── Panel C: Hessian curvature ────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])

    phases_c = ["pre_grok", "at_grok", "post_grok", "late"]
    phase_labels_c = ["Pre-grok", "At grok", "Post-grok", "Late"]

    for tag, ls, base_color in [("grok", "-", "#2166ac"), ("memo", "--", "#b2182b")]:
        for k, marker, alpha_val in [(0, "o", 1.0), (1, "s", 0.6), (2, "^", 1.0), (3, "D", 0.6)]:
            vals = []
            for p in phases_c:
                r = ablation.get(tag, {}).get(p, {})
                d = r.get("directions", {}).get(f"v{k+1}", {})
                vals.append(d.get("hessian_curvature", np.nan))

            if k < 2:
                color = base_color
                label_prefix = "edge"
            else:
                color = "#d6604d" if tag == "grok" else "#f4a582"
                label_prefix = "bulk"

            label = f"{tag} v{k+1} ({label_prefix})" if tag == "grok" else None
            ax_c.plot(range(len(phases_c)), vals, f'{marker}{ls}', color=color,
                     markersize=7, alpha=alpha_val, linewidth=1.5, label=label)

    ax_c.set_xticks(range(len(phases_c)))
    ax_c.set_xticklabels(phase_labels_c, fontsize=8)
    ax_c.set_ylabel("Directional Hessian curvature  v$^T$Hv", fontsize=10)
    ax_c.set_yscale("symlog", linthresh=0.05)
    ax_c.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax_c.axhline(y=0, ls="-", color="lightgray", lw=0.5)

    # Annotation regions
    ax_c.axhspan(-0.05, 0.1, alpha=0.08, color="blue", zorder=0)
    ax_c.text(3.4, 0.02, "flat\n(compression)", fontsize=7, ha="right", va="center",
              color="#2166ac", style="italic")
    ax_c.axhspan(0.5, 2.5, alpha=0.08, color="red", zorder=0)
    ax_c.text(3.4, 1.2, "curved\n(fragile)", fontsize=7, ha="right", va="center",
              color="#b2182b", style="italic")

    ax_c.set_title("C.  Loss Landscape: Flat Edge vs Curved Bulk", fontsize=11, fontweight="bold", pad=10)

    # ── Save ──
    fig.savefig(FIG_DIR / "paper_figure_thesis_connection.png", dpi=250)
    fig.savefig(FIG_DIR / "paper_figure_thesis_connection.pdf")
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'paper_figure_thesis_connection.png'}")
    print(f"Saved: {FIG_DIR / 'paper_figure_thesis_connection.pdf'}")


if __name__ == "__main__":
    main()
