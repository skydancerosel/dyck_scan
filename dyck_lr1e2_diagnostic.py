#!/usr/bin/env python3
"""
Diagnostic plot for Dyck lr=1e-2: raw acc, smoothed acc, defect.
Helps understand why lr=1e-2 groks slower than lr=1e-3 (non-monotonic).
"""

import sys, gc
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent / "dyck_pca_plots"
ACC_KEY = "test_acc"

# Load data
lr_sweep_path = OUT_DIR / "dyck_lr_sweep_results.pt"
cached = torch.load(lr_sweep_path, map_location="cpu", weights_only=False)

target_lr = 1e-2
seeds = [42, 137, 2024]

fig, axes = plt.subplots(len(seeds), 1, figsize=(14, 4*len(seeds)), sharex=False)

for idx, seed in enumerate(seeds):
    key = (target_lr, 1.0, seed)
    if key not in cached["all_runs"]:
        print(f"  Key {key} not found!")
        continue

    val = cached["all_runs"][key]
    recs = val.get("records", [])

    steps = np.array([r["step"] for r in recs])
    raw_acc = np.array([r.get(ACC_KEY, 0) for r in recs])
    defect = np.array([r.get("defect_median", 0) for r in recs])

    # Smoothed accuracy with different windows
    def smooth(arr, window):
        if len(arr) < window:
            return arr
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode='same')

    # Figure out step spacing to choose sensible smoothing windows
    if len(steps) > 1:
        step_spacing = np.median(np.diff(steps))
    else:
        step_spacing = 1

    # Window in number of records (not steps)
    # User asked for "window 200-500" in steps
    # Convert step-window to record-window
    win_200 = max(1, int(round(200 / step_spacing)))
    win_500 = max(1, int(round(500 / step_spacing)))

    smooth_200 = smooth(raw_acc, win_200)
    smooth_500 = smooth(raw_acc, win_500)

    ax = axes[idx]
    ax2 = ax.twinx()

    # Raw accuracy
    ax.plot(steps, raw_acc, color='#aaaaaa', linewidth=0.5, alpha=0.7, label='raw acc')
    # Smoothed accuracy
    ax.plot(steps, smooth_200, color='#1f77b4', linewidth=1.5, alpha=0.9,
            label=f'smooth acc (win={win_200} recs = ~{win_200*step_spacing:.0f} steps)')
    ax.plot(steps, smooth_500, color='#d62728', linewidth=2, alpha=0.9,
            label=f'smooth acc (win={win_500} recs = ~{win_500*step_spacing:.0f} steps)')

    # Horizontal line at 0.98
    ax.axhline(y=0.98, color='green', linestyle=':', alpha=0.5, linewidth=1, label='0.98 threshold')

    # Defect on secondary axis
    ax2.plot(steps, defect, color='#ff7f0e', linewidth=1.5, alpha=0.7, label='defect median')
    ax2.set_yscale('log')
    ax2.set_ylabel('Defect (log)', fontsize=10, color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')

    # Find grok step with sustained=3
    count = 0
    first_step = None
    grok_step = None
    for r in recs:
        if r.get(ACC_KEY, 0) >= 0.98:
            if count == 0:
                first_step = r["step"]
            count += 1
            if count >= 3:
                grok_step = first_step
                break
        else:
            count = 0
            first_step = None

    if grok_step is not None:
        ax.axvline(x=grok_step, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
                   label=f'grok @ {grok_step} (sustained=3)')

    # Also mark where acc first reaches 0.98 (without sustained criterion)
    first_98 = None
    for r in recs:
        if r.get(ACC_KEY, 0) >= 0.98:
            first_98 = r["step"]
            break
    if first_98 is not None and first_98 != grok_step:
        ax.axvline(x=first_98, color='purple', linestyle=':', alpha=0.5, linewidth=1.5,
                   label=f'first >= 0.98 @ {first_98}')

    ax.set_ylabel('Test Accuracy', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'Dyck lr=1e-2, seed={seed} | grok_step={grok_step} | '
                 f'{len(recs)} records, step_spacing={step_spacing:.0f}',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.2)

    # Print stats
    n_above = sum(1 for a in raw_acc if a >= 0.98)
    n_total = len(raw_acc)
    print(f"  seed={seed}: {n_total} records, step_spacing={step_spacing:.0f}, "
          f"acc>=0.98 in {n_above}/{n_total} records, "
          f"first_98={first_98}, grok(sustained=3)={grok_step}")
    print(f"    max_acc={max(raw_acc):.4f}, min_acc={min(raw_acc):.4f}, "
          f"final_acc={raw_acc[-1]:.4f}")

    # Check oscillation: count transitions above/below 0.98
    transitions = 0
    for i in range(1, len(raw_acc)):
        if (raw_acc[i] >= 0.98) != (raw_acc[i-1] >= 0.98):
            transitions += 1
    print(f"    transitions across 0.98: {transitions}")

fig.suptitle('Dyck lr=1e-2 Diagnostic: Raw Acc, Smoothed Acc, Defect',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "figDIAG_dyck_lr1e-2_grok_detection.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {OUT_DIR / 'figDIAG_dyck_lr1e-2_grok_detection.png'}")
