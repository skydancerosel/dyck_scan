#!/usr/bin/env python3
"""
Extended Dyck LR sweep: add lr={3e-5, 5e-5} with 3 seeds.

The existing lr={1e-4, 5e-4, 1e-3, 3e-3, 1e-2} data will be preserved.
"""

import dyck_lr_sweep

# Extend the grid with slower LRs
dyck_lr_sweep.LRS = [3e-5, 5e-5, 1e-4, 5e-4, 1e-3, 3e-3, 1e-2]
dyck_lr_sweep.SEEDS = [42, 137, 2024]

# Add max_steps and comm_every for new LR values
dyck_lr_sweep.MAX_STEPS_BY_LR.update({
    3e-5: 300_000,
    5e-5: 200_000,
})
dyck_lr_sweep.COMM_EVERY_BY_LR.update({
    3e-5: 1000,
    5e-5: 500,
})

if __name__ == "__main__":
    print("="*60)
    print("EXTENDED DYCK LR SWEEP")
    print(f"  LRs: {dyck_lr_sweep.LRS}")
    print(f"  Seeds: {dyck_lr_sweep.SEEDS}")
    print(f"  Total grid: {len(dyck_lr_sweep.LRS) * len(dyck_lr_sweep.SEEDS)} runs")
    print("="*60)

    all_runs = dyck_lr_sweep.run_sweep()
    dyck_lr_sweep.print_summary_table(all_runs)
    print("\nDone! Now re-run dyck_pc1_lr_experiment.py to regenerate figures.")
