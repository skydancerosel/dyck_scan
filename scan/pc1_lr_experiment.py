#!/usr/bin/env python3
"""
LR scaling law analysis for SCAN.

Uses existing data from scan_lr_sweep_results.pt and scan_sweep_results/
to derive defect onset lead time scaling law.

Tests whether SCAN exhibits similar super-linear scaling as Dyck:
  lead_time ~ grok_step^alpha, with alpha > 1
  => predictive window IMPROVES at slower learning rates.

Also computes PC1 trajectories from existing weight snapshots (lr=1e-4 only).
"""

import sys, gc
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from scan_generalization_dynamics import find_spike_step
from scan_grok_sweep import extract_attn_matrices

# -- config -------------------------------------------------------------------
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42  # primary seed for scaling analysis


# -- PC1 computation ----------------------------------------------------------
def compute_pc1_from_logs(attn_logs, subsample=1):
    """Compute expanding-window PC1 variance ratio from weight logs."""
    logs = attn_logs[::subsample] if subsample > 1 else attn_logs
    T = len(logs)

    KEYS = ['WQ', 'WK', 'WV', 'WO', 'XWQ', 'XWK', 'XWV', 'XWO', 'W_up', 'W_down']

    def flatten_snapshot(layers):
        parts = []
        for layer in layers:
            for key in KEYS:
                if key in layer:
                    w = layer[key]
                    if isinstance(w, torch.Tensor):
                        w = w.numpy()
                    parts.append(w.flatten())
        return np.concatenate(parts)

    w0 = flatten_snapshot(logs[0]['layers'])
    D = len(w0)

    all_deltas = np.zeros((T, D))
    steps = []
    for t in range(T):
        all_deltas[t] = flatten_snapshot(logs[t]['layers']) - w0
        steps.append(logs[t]['step'])

    results = []
    for t in range(3, T):
        traj = all_deltas[:t+1]
        traj_c = traj - traj.mean(axis=0)
        gram = traj_c @ traj_c.T
        eigvals = np.linalg.eigvalsh(gram)[::-1]
        total = eigvals.sum()
        pc1 = eigvals[0] / total if total > 1e-12 else 1.0
        results.append({'step': steps[t], 'pc1': pc1})

    return results


# -- Analysis helpers ----------------------------------------------------------
def find_defect_onset(records, threshold_factor=10, min_defect=20):
    """Find first step where defect spikes above baseline."""
    return find_spike_step(records, threshold_factor=threshold_factor,
                           min_defect=min_defect)


def find_grok_step_acc(records, acc_threshold=0.98, sustained=3):
    """Find first step where test_seq_acc >= threshold for `sustained` consecutive evals."""
    count = 0
    first_step = None
    for r in records:
        if r.get("test_seq_acc", 0) >= acc_threshold:
            if count == 0:
                first_step = r["step"]
            count += 1
            if count >= sustained:
                return first_step
        else:
            count = 0
            first_step = None
    return None


def find_pc1_turnover(pc1_data):
    """Find step where PC1% starts sustained decrease (de-concentration)."""
    if len(pc1_data) < 5:
        return None, None, None

    max_pc1 = max(r['pc1'] for r in pc1_data)
    max_step = [r['step'] for r in pc1_data if r['pc1'] == max_pc1][0]
    min_pc1 = min(r['pc1'] for r in pc1_data)
    min_step = [r['step'] for r in pc1_data if r['pc1'] == min_pc1][0]

    turnover_step = None
    for i in range(2, len(pc1_data)):
        if pc1_data[i]['step'] > max_step:
            if (pc1_data[i]['pc1'] < pc1_data[i-1]['pc1'] < pc1_data[i-2]['pc1']):
                turnover_step = pc1_data[i-2]['step']
                break

    return max_step, min_step, turnover_step


# -- Load all existing data ----------------------------------------------------
def load_all_data():
    """Load defect data from lr_sweep and weight data from grok_sweep."""
    all_data = {}

    # 1. Load lr_sweep_results (has defect records at multiple LRs and seeds)
    lr_sweep_path = OUT_DIR / "scan_lr_sweep_results.pt"
    if lr_sweep_path.exists():
        cached = torch.load(lr_sweep_path, map_location="cpu", weights_only=False)
        if "all_runs" in cached:
            for key, val in cached["all_runs"].items():
                lr, wd, seed = key
                if wd != 1.0:
                    continue
                all_data[(lr, seed)] = val
        del cached; gc.collect()
        print(f"  Loaded lr_sweep data: {len(all_data)} runs")

    # 2. Load weight snapshots from scan_sweep_results/ (lr=1e-4 only)
    sweep_dir = Path(__file__).parent / "../scan_sweep_results"
    for seed in [42, 137, 2024]:
        fpath = sweep_dir / f"scan_wd1.0_s{seed}.pt"
        if fpath.exists():
            data = torch.load(fpath, map_location="cpu", weights_only=False)
            key = (1e-4, seed)
            if key in all_data:
                all_data[key]["attn_logs"] = data["attn_logs"]
            else:
                all_data[key] = {
                    "attn_logs": data["attn_logs"],
                    "records": data.get("metrics", []),
                    "grokked": data.get("grokked", False),
                    "grok_step": data.get("final_step") if data.get("grokked") else None,
                }
            del data; gc.collect()

    return all_data


# -- Figures -------------------------------------------------------------------
def plot_lead_scaling(scaling_data):
    """3-panel scaling figure: lead vs LR, lead fraction vs LR, onset vs grok."""
    if len(scaling_data) < 2:
        print(f"  SKIP scaling figure: only {len(scaling_data)} usable points")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    lrs = [d["lr"] for d in scaling_data]
    leads = [d["lead_time"] for d in scaling_data]
    fracs = [d["lead_fraction"] for d in scaling_data]
    groks = [d["grok_step"] for d in scaling_data]
    onsets = [d["onset_step"] for d in scaling_data]

    # (A) Lead time vs LR
    ax = axes[0]
    ax.scatter(lrs, leads, s=100, c='#d62728', zorder=5)
    for d in scaling_data:
        ax.annotate(f"  {d['lead_time']:.0f}", (d["lr"], d["lead_time"]),
                    fontsize=9, ha='left')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Learning rate", fontsize=11)
    ax.set_ylabel("Lead time (steps)", fontsize=11)
    ax.set_title("(A) Lead time vs LR")
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    # (B) Lead fraction vs LR
    ax = axes[1]
    ax.scatter(lrs, fracs, s=100, c='#2ca02c', zorder=5)
    for d in scaling_data:
        ax.annotate(f"  {d['lead_fraction']:.1%}", (d["lr"], d["lead_fraction"]),
                    fontsize=9, ha='left')
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate", fontsize=11)
    ax.set_ylabel("Lead fraction (lead / grok_step)", fontsize=11)
    ax.set_title("(B) Lead fraction vs LR")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    # (C) Onset step vs Grok step with power-law fit
    ax = axes[2]
    ax.scatter(groks, onsets, s=100, c='#1f77b4', zorder=5)
    for d in scaling_data:
        ax.annotate(f"  lr={d['lr']:.0e}", (d["grok_step"], d["onset_step"]),
                    fontsize=8, ha='left')

    if len(groks) >= 2:
        log_g = np.log(np.array(groks, dtype=float))
        log_o = np.log(np.array(onsets, dtype=float))
        slope, intercept, r, p, se = stats.linregress(log_g, log_o)
        C = np.exp(intercept)

        g_fit = np.logspace(np.log10(min(groks)*0.5), np.log10(max(groks)*1.5), 100)
        o_fit = C * g_fit ** slope
        ax.plot(g_fit, o_fit, 'k--', alpha=0.6, linewidth=1.5,
                label=f"onset ~ grok^{slope:.2f} (R²={r**2:.3f})")
        ax.plot(g_fit, g_fit, ':', color='gray', alpha=0.4,
                label='onset = grok (no lead)')

        # Lead time power law
        log_leads = np.log(np.array([max(l, 1) for l in leads], dtype=float))
        slope2, intercept2, r2, p2, se2 = stats.linregress(log_g, log_leads)

        ax.set_title(f"(C) Onset vs Grok step\n"
                     f"lead ~ grok^{slope2:.2f}, onset ~ grok^{slope:.2f}")
        ax.legend(fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Grok step", fontsize=11)
    ax.set_ylabel("Defect onset step", fontsize=11)
    ax.grid(alpha=0.3)

    fig.suptitle("SCAN: Defect Onset Lead Time Scaling Law",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPC1_scan_lr_lead_scaling.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figPC1_scan_lr_lead_scaling.png")


def plot_lead_scaling_multiseed(all_data):
    """Lead time scaling using all seeds, showing individual points + mean."""
    all_lrs = sorted(set(k[0] for k in all_data.keys()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    all_points = []
    mean_by_lr = {}

    for lr in all_lrs:
        lr_points = []
        for key, val in all_data.items():
            if key[0] != lr:
                continue
            recs = val.get("records", [])
            if not recs:
                continue

            grok_step = find_grok_step_acc(recs, acc_threshold=0.98, sustained=3)
            if grok_step is None:
                continue

            onset = find_defect_onset(recs)
            if onset is None:
                continue

            lead = grok_step - onset
            if lead <= 0:
                continue
            frac = lead / grok_step

            pt = {"lr": lr, "seed": key[1], "grok_step": grok_step,
                  "onset_step": onset, "lead_time": lead, "lead_fraction": frac}
            all_points.append(pt)
            lr_points.append(pt)

        if lr_points:
            mean_by_lr[lr] = {
                "grok_step": np.mean([p["grok_step"] for p in lr_points]),
                "onset_step": np.mean([p["onset_step"] for p in lr_points]),
                "lead_time": np.mean([p["lead_time"] for p in lr_points]),
                "lead_fraction": np.mean([p["lead_fraction"] for p in lr_points]),
                "n": len(lr_points),
            }

    if len(all_points) < 2:
        print(f"  SKIP multiseed scaling: only {len(all_points)} points")
        return all_points, mean_by_lr

    # Scatter all individual points
    colors = {'#1f77b4': 42, '#ff7f0e': 137, '#2ca02c': 2024}
    seed_colors = {42: '#1f77b4', 137: '#ff7f0e', 2024: '#2ca02c'}

    # (A) Lead time vs LR - individual seeds
    ax = axes[0]
    for pt in all_points:
        c = seed_colors.get(pt["seed"], "gray")
        ax.scatter(pt["lr"], pt["lead_time"], s=60, c=c, alpha=0.6, zorder=4)
    # Mean line
    mean_lrs = sorted(mean_by_lr.keys())
    ax.plot([lr for lr in mean_lrs],
            [mean_by_lr[lr]["lead_time"] for lr in mean_lrs],
            'k-o', markersize=8, linewidth=2, zorder=5, label="mean")
    for lr in mean_lrs:
        m = mean_by_lr[lr]
        ax.annotate(f"  {m['lead_time']:.0f} (n={m['n']})",
                    (lr, m["lead_time"]), fontsize=8, ha='left')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Learning rate", fontsize=11)
    ax.set_ylabel("Lead time (steps)", fontsize=11)
    ax.set_title("(A) Lead time vs LR")
    ax.grid(alpha=0.3)
    ax.invert_xaxis()
    ax.legend(fontsize=8)

    # (B) Lead fraction vs LR
    ax = axes[1]
    for pt in all_points:
        c = seed_colors.get(pt["seed"], "gray")
        ax.scatter(pt["lr"], pt["lead_fraction"], s=60, c=c, alpha=0.6, zorder=4)
    ax.plot([lr for lr in mean_lrs],
            [mean_by_lr[lr]["lead_fraction"] for lr in mean_lrs],
            'k-o', markersize=8, linewidth=2, zorder=5, label="mean")
    for lr in mean_lrs:
        m = mean_by_lr[lr]
        ax.annotate(f"  {m['lead_fraction']:.1%}",
                    (lr, m["lead_fraction"]), fontsize=8, ha='left')
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate", fontsize=11)
    ax.set_ylabel("Lead fraction (lead / grok_step)", fontsize=11)
    ax.set_title("(B) Lead fraction vs LR")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()
    ax.legend(fontsize=8)

    # (C) Onset vs grok with power-law fit
    ax = axes[2]
    groks = np.array([p["grok_step"] for p in all_points], dtype=float)
    onsets = np.array([p["onset_step"] for p in all_points], dtype=float)
    leads = np.array([p["lead_time"] for p in all_points], dtype=float)

    for pt in all_points:
        c = seed_colors.get(pt["seed"], "gray")
        ax.scatter(pt["grok_step"], pt["onset_step"], s=60, c=c, alpha=0.6, zorder=4)
        ax.annotate(f" {pt['lr']:.0e}", (pt["grok_step"], pt["onset_step"]),
                    fontsize=7, ha='left', alpha=0.7)

    # Power-law fit on means
    if len(mean_lrs) >= 2:
        m_groks = np.array([mean_by_lr[lr]["grok_step"] for lr in mean_lrs])
        m_onsets = np.array([mean_by_lr[lr]["onset_step"] for lr in mean_lrs])
        m_leads = np.array([mean_by_lr[lr]["lead_time"] for lr in mean_lrs])

        log_g = np.log(m_groks)
        log_o = np.log(m_onsets)
        slope, intercept, r, p, se = stats.linregress(log_g, log_o)

        g_fit = np.logspace(np.log10(min(groks)*0.5), np.log10(max(groks)*1.5), 100)
        o_fit = np.exp(intercept) * g_fit ** slope
        ax.plot(g_fit, o_fit, 'k--', alpha=0.6, linewidth=1.5,
                label=f"onset ~ grok^{slope:.2f} (R²={r**2:.3f})")
        ax.plot(g_fit, g_fit, ':', color='gray', alpha=0.4,
                label='onset = grok (no lead)')

        log_l = np.log(m_leads)
        slope2, intercept2, r2, p2, se2 = stats.linregress(log_g, log_l)
        ax.set_title(f"(C) Onset vs Grok step (means)\n"
                     f"lead ~ grok^{slope2:.2f}, onset ~ grok^{slope:.2f}")
        ax.legend(fontsize=8)

    # Mean points
    ax.scatter([mean_by_lr[lr]["grok_step"] for lr in mean_lrs],
               [mean_by_lr[lr]["onset_step"] for lr in mean_lrs],
               s=120, c='black', marker='D', zorder=6, label="LR mean")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Grok step", fontsize=11)
    ax.set_ylabel("Defect onset step", fontsize=11)
    ax.grid(alpha=0.3)

    fig.suptitle("SCAN: Defect Onset Lead Time Scaling (all seeds)",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPC1_scan_lr_lead_scaling_multiseed.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figPC1_scan_lr_lead_scaling_multiseed.png")

    return all_points, mean_by_lr


def plot_hero(all_data):
    """Hero figure: defect + test_acc at each LR for seed=42."""
    all_lrs = sorted(set(k[0] for k in all_data.keys() if k[1] == SEED))
    LR_COLORS = {}
    cmap_vals = plt.cm.viridis(np.linspace(0.15, 0.85, len(all_lrs)))
    for i, lr in enumerate(all_lrs):
        LR_COLORS[lr] = cmap_vals[i]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()

    for lr in all_lrs:
        key = (lr, SEED)
        if key not in all_data:
            continue
        recs = all_data[key].get("records", [])
        if not recs:
            continue

        color = LR_COLORS[lr]
        steps = [r["step"] for r in recs]
        defects = [r["defect_median"] for r in recs]
        test_accs = [r.get("test_seq_acc", 0) for r in recs]

        ax.plot(steps, defects, color=color, linewidth=2, alpha=0.85)
        ax2.plot(steps, test_accs, color=color, linewidth=1.5, linestyle="--",
                 alpha=0.6)

        onset = find_defect_onset(recs)
        grok = find_grok_step_acc(recs, 0.98, sustained=3)
        if onset is not None:
            ax.axvline(x=onset, color=color, linestyle=":", alpha=0.5, linewidth=1)
        if grok is not None:
            ax.axvline(x=grok, color=color, linestyle="-.", alpha=0.3, linewidth=1)

    ax.set_yscale("log")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Commutator defect", fontsize=11)
    ax2.set_ylabel("Test seq accuracy", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax.set_title("SCAN: Defect Predicts Grokking across Learning Rates",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.2)

    handles = []
    for lr in all_lrs:
        c = LR_COLORS[lr]
        handles.append(Line2D([0], [0], color=c, linewidth=2,
                              label=f"lr={lr:.0e} defect"))
        handles.append(Line2D([0], [0], color=c, linewidth=1.5, linestyle="--",
                              label=f"lr={lr:.0e} test acc"))
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPD2_scan_pc1_lr_hero.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figPD2_scan_pc1_lr_hero.png")


def plot_pc1_trajectory(attn_logs, grok_step=None, lr=None, seed=None):
    """Plot PC1 trajectory for a single run with weight snapshots."""
    n = len(attn_logs)
    sub = max(1, n // 80)
    print(f"  Computing PC1 (seed={seed}): {n} snapshots (subsample={sub})...",
          end=" ", flush=True)
    pc1 = compute_pc1_from_logs(attn_logs, subsample=sub)
    print(f"done ({len(pc1)} points)")

    if len(pc1) < 5:
        print(f"  SKIP PC1 plot: too few points")
        return pc1

    fig, ax = plt.subplots(figsize=(8, 4))
    steps = [r['step'] for r in pc1]
    vals = [r['pc1'] for r in pc1]
    ax.plot(steps, vals, 'b-', linewidth=2)

    if grok_step:
        ax.axvline(x=grok_step, color='red', linestyle='--', alpha=0.7,
                   label=f'Grok @ {grok_step}')

    max_step, min_step, turnover = find_pc1_turnover(pc1)
    if turnover:
        ax.axvline(x=turnover, color='green', linestyle=':', alpha=0.7,
                   label=f'PC1 turnover @ {turnover}')
    if grok_step and turnover:
        lead = grok_step - turnover
        ax.set_title(f"SCAN PC1 Trajectory (lr={lr:.0e}, seed={seed}, lead={lead})")
    else:
        ax.set_title(f"SCAN PC1 Trajectory (lr={lr:.0e}, seed={seed})")

    ax.set_xlabel("Training step")
    ax.set_ylabel("PC1 explained variance")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fname = f"figPC1_scan_trajectory_lr{lr:.0e}_s{seed}.png"
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")
    return pc1


# -- Main ---------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("SCAN: LR Scaling Law Analysis")
    print("="*60)

    # Load all existing data
    all_data = load_all_data()

    print(f"\n  Available data:")
    for key in sorted(all_data.keys()):
        lr, seed = key
        val = all_data[key]
        recs = val.get("records", [])
        has_weights = "attn_logs" in val and bool(val.get("attn_logs"))
        grokked = val.get("grokked", False)
        grok_step = val.get("grok_step")
        print(f"    lr={lr:.0e} seed={seed}: {len(recs)} records, "
              f"grokked={grokked}, grok_step={grok_step}, weights={has_weights}")

    # -- Defect-based scaling analysis (all seeds) --
    print(f"\n{'='*60}")
    print("DEFECT ONSET SCALING ANALYSIS")
    print(f"{'='*60}")

    print(f"\n  Per-run analysis (grok defined as first test_seq_acc >= 0.98):")
    print(f"  {'LR':>8s}  {'Seed':>5s}  {'GrokStep':>10s}  {'Onset':>8s}  "
          f"{'Lead':>8s}  {'Frac':>8s}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")

    scaling_points = []
    for key in sorted(all_data.keys()):
        lr, seed = key
        val = all_data[key]
        recs = val.get("records", [])
        if not recs:
            continue

        # Always use consistent sustained-accuracy criterion
        grok_step = find_grok_step_acc(recs, acc_threshold=0.98, sustained=3)
        if grok_step is None:
            print(f"  {lr:>8.0e}  {seed:>5d}  {'---':>10s}  {'---':>8s}  "
                  f"{'---':>8s}  {'---':>8s}  (no grok)")
            continue

        onset = find_defect_onset(recs)
        if onset is None:
            print(f"  {lr:>8.0e}  {seed:>5d}  {grok_step:>10d}  {'---':>8s}  "
                  f"{'---':>8s}  {'---':>8s}  (no onset)")
            continue

        lead = grok_step - onset
        if lead <= 0:
            print(f"  {lr:>8.0e}  {seed:>5d}  {grok_step:>10d}  {onset:>8d}  "
                  f"{lead:>8d}  {'---':>8s}  (negative lead)")
            continue

        frac = lead / grok_step
        scaling_points.append({
            "lr": lr, "seed": seed, "grok_step": grok_step,
            "onset_step": onset, "lead_time": lead, "lead_fraction": frac,
        })
        print(f"  {lr:>8.0e}  {seed:>5d}  {grok_step:>10d}  {onset:>8d}  "
              f"{lead:>8d}  {frac:>8.1%}")

    # -- Compute scaling law fit --
    if len(scaling_points) >= 2:
        groks = np.array([p["grok_step"] for p in scaling_points], dtype=float)
        leads = np.array([p["lead_time"] for p in scaling_points], dtype=float)
        fracs = np.array([p["lead_fraction"] for p in scaling_points], dtype=float)

        # Fit: lead_time = A * grok_step^beta
        log_g = np.log(groks)
        log_l = np.log(leads)
        beta, log_A, r, p, se = stats.linregress(log_g, log_l)
        A = np.exp(log_A)

        print(f"\n  Power-law fit (all {len(scaling_points)} points):")
        print(f"    lead_time = {A:.4f} * grok_step ^ {beta:.3f}")
        print(f"    R² = {r**2:.4f}, p = {p:.6f}, SE(beta) = {se:.3f}")

        # Also fit by LR means
        lr_groups = {}
        for pt in scaling_points:
            lr_groups.setdefault(pt["lr"], []).append(pt)

        mean_lrs = sorted(lr_groups.keys())
        if len(mean_lrs) >= 2:
            m_groks = [np.mean([p["grok_step"] for p in lr_groups[lr]])
                       for lr in mean_lrs]
            m_leads = [np.mean([p["lead_time"] for p in lr_groups[lr]])
                       for lr in mean_lrs]
            m_fracs = [np.mean([p["lead_fraction"] for p in lr_groups[lr]])
                       for lr in mean_lrs]

            log_mg = np.log(np.array(m_groks))
            log_ml = np.log(np.array(m_leads))
            beta_m, log_Am, rm, pm, sem = stats.linregress(log_mg, log_ml)

            print(f"\n  Power-law fit (LR means, {len(mean_lrs)} points):")
            print(f"    lead_time = {np.exp(log_Am):.4f} * grok_step ^ {beta_m:.3f}")
            print(f"    R² = {rm**2:.4f}")

            print(f"\n  Per-LR summary:")
            print(f"  {'LR':>8s}  {'N':>3s}  {'MeanGrok':>10s}  {'MeanLead':>10s}  {'MeanFrac':>10s}")
            for i, lr in enumerate(mean_lrs):
                print(f"  {lr:>8.0e}  {len(lr_groups[lr]):>3d}  {m_groks[i]:>10.0f}  "
                      f"{m_leads[i]:>10.0f}  {m_fracs[i]:>10.1%}")

    # -- PC1 trajectory for lr=1e-4 (has weight snapshots) --
    print(f"\n{'='*60}")
    print("PC1 TRAJECTORY (lr=1e-4, weight snapshots available)")
    print(f"{'='*60}")

    pc1_data = {}
    for seed in [42, 137, 2024]:
        key = (1e-4, seed)
        if key in all_data and "attn_logs" in all_data[key] and all_data[key]["attn_logs"]:
            recs = all_data[key].get("records", [])
            grok_step = find_grok_step_acc(recs, 0.98, sustained=3) if recs else None
            pc1 = plot_pc1_trajectory(all_data[key]["attn_logs"],
                                      grok_step=grok_step, lr=1e-4, seed=seed)
            pc1_data[(1e-4, seed)] = pc1

    # -- Generate scaling figures --
    print(f"\n  Generating scaling figures...")

    # Single-seed scaling (seed=42 only)
    seed42_points = [p for p in scaling_points if p["seed"] == SEED]
    plot_lead_scaling(seed42_points)

    # Multi-seed scaling
    all_pts, mean_by_lr = plot_lead_scaling_multiseed(all_data)

    # Hero figure
    plot_hero(all_data)

    # -- Final summary --
    if len(scaling_points) >= 2:
        groks = np.array([p["grok_step"] for p in scaling_points], dtype=float)
        leads = np.array([p["lead_time"] for p in scaling_points], dtype=float)
        log_g = np.log(groks)
        log_l = np.log(leads)
        beta, _, r, p, se = stats.linregress(log_g, log_l)

        print(f"\n{'='*60}")
        print("SCALING LAW RESULT")
        print(f"{'='*60}")
        print(f"  lead_time ~ grok_step ^ {beta:.2f}  (R²={r**2:.3f})")

        slowest = min(scaling_points, key=lambda x: x["lr"])
        fastest = max(scaling_points, key=lambda x: x["lr"])
        print(f"  Slowest (lr={slowest['lr']:.0e}): "
              f"lead = {slowest['lead_fraction']:.1%} of training")
        print(f"  Fastest (lr={fastest['lr']:.0e}): "
              f"lead = {fastest['lead_fraction']:.1%} of training")

        if beta > 1:
            print(f"\n  SUPER-LINEAR (alpha={beta:.2f} > 1): "
                  f"predictive window IMPROVES at slower LRs!")
        elif beta < 1:
            print(f"\n  SUB-LINEAR (alpha={beta:.2f} < 1)")
        else:
            print(f"\n  LINEAR (alpha ~ 1)")

        comparison = "similar to" if abs(beta - 1.3) < 0.3 else "different from"
        print(f"  Dyck result: alpha ~ 1.3; SCAN: alpha ~ {beta:.2f} ({comparison} Dyck)")

    print(f"\n  Done!")
