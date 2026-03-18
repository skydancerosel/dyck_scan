#!/usr/bin/env python3
"""
Extended SCAN LR sweep: add seeds {137, 2024} at lr={1e-5, 5e-5, 5e-4, 1e-3}
and add lr={5e-4, 1e-3} for seed=42 if missing.

This script monkey-patches scan_lr_sweep to extend the grid, then runs it.
The existing cached results will be preserved (scan_lr_sweep skips cached runs).

Measurement frequencies tuned per LR — denser at fast LRs where onset is early:
  lr=1e-3: onset~800, grok~29k  → comm every 500, max 60k
  lr=5e-4: onset~1000, grok~21k → comm every 500, max 30k
  lr=1e-4: onset~700, grok~7k   → comm every 500, max 30k
  lr=5e-5: onset~1000, grok~28k → comm every 1000, max 50k
  lr=1e-5: onset~3000, grok~128k→ comm every 2000, max 200k
"""

import scan_lr_sweep

# Extend the grid — fastest LRs first
scan_lr_sweep.LRS = [1e-3, 5e-4, 1e-4, 5e-5]  # skip 1e-5 (1 seed is enough)
scan_lr_sweep.SEEDS = [42, 137, 2024]

# Max steps: generous budget to allow seed variance
scan_lr_sweep.MAX_STEPS_BY_LR.update({
    1e-3: 60_000,
    5e-4: 30_000,
    1e-4: 30_000,
    5e-5: 50_000,
    1e-5: 200_000,
})
scan_lr_sweep.COMM_EVERY_BY_LR.update({
    1e-3: 500,     # dense: onset at ~800 steps
    5e-4: 500,
    1e-4: 500,
    5e-5: 1000,
    1e-5: 2000,
})

# Reduce number of defect measurements per checkpoint (3 instead of 5)
scan_lr_sweep.COMM_K = 3

if __name__ == "__main__":
    print("="*60)
    print("EXTENDED SCAN LR SWEEP")
    print(f"  LRs: {scan_lr_sweep.LRS}")
    print(f"  Seeds: {scan_lr_sweep.SEEDS}")
    print(f"  Total grid: {len(scan_lr_sweep.LRS) * len(scan_lr_sweep.SEEDS)} runs")
    print("="*60)

    all_runs = scan_lr_sweep.run_sweep()
    scan_lr_sweep.print_summary_table(all_runs)
    print("\nDone! Now re-run scan_pc1_lr_experiment.py to regenerate figures.")
