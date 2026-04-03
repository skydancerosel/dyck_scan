#!/usr/bin/env python3
"""
Spectral Edge Thesis – Table 7 replication for Dyck and SCAN datasets.

Computes three spectral quantities from training trajectories following
the methodology of Xu (2026), arXiv:2603.28964:

  1. g₂₃ = λ₂ − λ₃ = σ₂² − σ₃²   (eigenvalue gap of Gram matrix)
  2. R = σ_{k*} / σ_{k*+1}          (gap ratio at weighted k*)
  3. Weighted k*                      (signal-weighted argmax)

All quantities derived from the rolling-window Gram matrix:
  G(t) = X(t) X(t)^T  where  X(t) = [δ_{t-W+1}, ..., δ_t]^T ∈ ℝ^{W×p}
  δ_t = θ_t − θ_{t-1}  (all W* from all layers flattened into ℝ^p)

Usage:
    python3 -u spectral/thesis_table7_replication.py
    MAX_FILE_MB=1100 python3 -u spectral/thesis_table7_replication.py
"""

import os, sys, math, warnings
from pathlib import Path
from collections import OrderedDict, Counter

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent
MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", 200))

GRAM_WINDOW_DYCK = 3    # Dyck groks early (step 600-1400), needs small window
GRAM_WINDOW_SCAN = 5    # SCAN groks later (step 2500-4000), can use larger window
GROK_THRESHOLD = 0.95   # test_acc threshold for grokking detection
MONOTONIC_TOL = 1       # allow ≤1 increase for "monotonic" classification

SEEDS = [42, 137, 2024]
WD_VALUES = [1.0, 0.0]

# W* keys to extract from each layer
ATTN_KEYS = ["WQ", "WK", "WV", "WO"]
XATTN_KEYS = ["XWQ", "XWK", "XWV", "XWO"]

# Dense checkpoint overrides: {(wd, seed): filename}
DYCK_DENSE_OVERRIDES = {
    (1.0, 42): "dyck_wd1.0_s42_dense.pt",
}

SCAN_DENSE_OVERRIDES = {
    (1.0, 2024): "scan_wd1.0_s2024_dense.pt",
}

DATASETS = OrderedDict([
    ("dyck", {
        "results_dir": ROOT / "dyck_sweep_results",
        "file_pattern": "dyck_wd{wd}_s{seed}.pt",
        "acc_key": "test_acc",
        "gram_window": GRAM_WINDOW_DYCK,
        "dense_overrides": DYCK_DENSE_OVERRIDES,
    }),
    ("scan", {
        "results_dir": ROOT / "scan_sweep_results",
        "file_pattern": "scan_wd{wd}_s{seed}.pt",
        "acc_key": "test_seq_acc",
        "gram_window": GRAM_WINDOW_SCAN,
        "dense_overrides": SCAN_DENSE_OVERRIDES,
    }),
])

OUTPUT_DIR = Path(__file__).resolve().parent / "coherence_edge_plots"
RESULTS_DIR = Path(__file__).resolve().parent / "coherence_edge_results"
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_run(path: Path):
    """Load a .pt checkpoint file, return None if too large."""
    size_mb = path.stat().st_size / 1e6
    if size_mb > MAX_FILE_MB:
        print(f"  SKIP {path.name} ({size_mb:.0f} MB > {MAX_FILE_MB} MB limit)")
        return None
    print(f"  Loading {path.name} ({size_mb:.0f} MB)...")
    return torch.load(path, map_location="cpu", weights_only=False)


def find_grok_step(metrics, threshold=GROK_THRESHOLD, acc_key="test_acc"):
    """Return the first step where acc_key >= threshold, or None."""
    for m in metrics:
        val = m.get(acc_key, m.get("test_acc", 0))
        if val >= threshold:
            return m["step"]
    return None


def subsample_attn_logs(attn_logs, grok_step):
    """Subsample attn_logs for large files: keep first 50 + 20 around grok + last 15."""
    n = len(attn_logs)
    if n <= 100:
        return attn_logs

    keep = set(range(min(50, n)))
    keep.update(range(max(0, n - 15), n))

    if grok_step is not None:
        steps = [e["step"] for e in attn_logs]
        grok_idx = min(range(n), key=lambda i: abs(steps[i] - grok_step))
        keep.update(range(max(0, grok_idx - 10), min(n, grok_idx + 10)))

    keep = sorted(keep)
    return [attn_logs[i] for i in keep]


def flatten_all_W(entry):
    """
    Flatten ALL W* matrices from ALL layers into a single vector ℝ^p.
    Includes self-attention (WQ,WK,WV,WO) and cross-attention (XWQ,...) if present.
    """
    parts = []
    for ld in entry["layers"]:
        for k in ATTN_KEYS + XATTN_KEYS:
            if k in ld:
                parts.append(ld[k].float().flatten())
    if not parts:
        return None
    return torch.cat(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Core: Gram matrix → g₂₃, R, k*
#
# All three quantities come from the SVD of X(t) ∈ ℝ^{W×p}:
#   σ₁ ≥ σ₂ ≥ ... ≥ σ_W   (singular values of X)
#   λ_j = σ_j²              (eigenvalues of G = X X^T)
#
#   g₂₃ = λ₂ − λ₃ = σ₂² − σ₃²
#   k*  = argmax_j (σ_j/Σσ)(σ_j/σ_{j+1})
#   R   = σ_{k*} / σ_{k*+1}
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_gram_quantities(attn_logs, W=5):
    """
    From the sequence of attn_log entries, compute rolling-window Gram
    matrix quantities at each step.

    δ_t = flatten(all W* at step t) - flatten(all W* at step t-1)
    X(t) = [δ_{t-w+1}, ..., δ_t]^T   (adaptive w = min(W, available deltas))
    σ = svdvals(X)

    Returns list of dicts:
        [{step, g23, R, kstar, sv, window_size}, ...]
    """
    # collect (step, θ_t) pairs
    param_series = []
    for entry in attn_logs:
        step = entry["step"]
        vec = flatten_all_W(entry)
        if vec is not None:
            param_series.append((step, vec))

    if len(param_series) < 3:
        return []

    # δ_t = θ_t - θ_{t-1}
    deltas = []
    for i in range(1, len(param_series)):
        step = param_series[i][0]
        delta = param_series[i][1] - param_series[i - 1][1]
        deltas.append((step, delta))

    results = []
    for i in range(1, len(deltas)):
        w = min(W, i + 1)  # adaptive window
        X = torch.stack([deltas[j][1] for j in range(i - w + 1, i + 1)])  # [w, p]

        sv = torch.linalg.svdvals(X.float()).numpy()

        # --- g₂₃ = σ₂² − σ₃² ---
        if len(sv) >= 3:
            g23 = float(sv[1] ** 2 - sv[2] ** 2)
        elif len(sv) >= 2:
            g23 = float(sv[1] ** 2)
        else:
            g23 = 0.0

        # --- weighted k* ---
        total = sv.sum()
        if total < 1e-12:
            results.append({"step": deltas[i][0], "g23": g23, "R": 1.0,
                            "kstar": 1, "sv": sv.tolist(), "window_size": w})
            continue

        best_score = -1.0
        best_j = 0
        for j in range(len(sv) - 1):
            if sv[j + 1] < 1e-12:
                score = (sv[j] / total) * 1e6
            else:
                score = (sv[j] / total) * (sv[j] / sv[j + 1])
            if score > best_score:
                best_score = score
                best_j = j

        kstar = best_j + 1  # 1-indexed

        # --- R = σ_{k*} / σ_{k*+1} ---
        if kstar < len(sv) and sv[kstar] > 1e-12:
            R = float(sv[kstar - 1] / sv[kstar])
        else:
            R = float(sv[kstar - 1] / 1e-12) if sv[kstar - 1] > 0 else 1.0

        results.append({
            "step": deltas[i][0],
            "g23": g23,
            "R": R,
            "kstar": int(kstar),
            "sv": sv.tolist(),
            "window_size": w,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Analysis: Table 7 quantities
# ═══════════════════════════════════════════════════════════════════════════

def analyze_run(data, dataset_cfg):
    """Compute all Table 7 quantities for a single run."""
    metrics = data["metrics"]
    attn_logs = data["attn_logs"]
    cfg = data["cfg"]
    wd = cfg.get("wd", cfg.get("weight_decay", 0.0))

    acc_key = dataset_cfg.get("acc_key", "test_acc")
    grok_step = find_grok_step(metrics, acc_key=acc_key)
    grokked = grok_step is not None

    attn_logs = subsample_attn_logs(attn_logs, grok_step)

    n_layers = len(attn_logs[0]["layers"]) if attn_logs else 0
    p = flatten_all_W(attn_logs[0]).numel() if attn_logs else 0
    W = dataset_cfg.get("gram_window", 5)

    result = {
        "wd": wd,
        "grokked": grokked,
        "grok_step": grok_step,
        "n_checkpoints": len(attn_logs),
        "n_layers": n_layers,
        "p_total": p,
        "gram_window": W,
    }

    # --- All quantities from Gram matrix ---
    gram = compute_all_gram_quantities(attn_logs, W=W)

    if not gram:
        result.update({"g23_early": None, "g23_grok": None, "decline": None,
                        "monotonic": None, "R_early": None, "R_grok": None,
                        "kstar_mode": None})
        return result

    g23_traj = [(r["step"], r["g23"]) for r in gram]
    R_traj = [(r["step"], r["R"]) for r in gram]
    kstar_vals = [r["kstar"] for r in gram]

    # --- g₂₃ analysis ---
    if grokked:
        pre_grok_g23 = [(s, v) for s, v in g23_traj if s <= grok_step]
        if pre_grok_g23:
            early_step, g23_early = max(pre_grok_g23, key=lambda x: x[1])
        else:
            early_step, g23_early = g23_traj[0]

        at_grok_g23 = [(s, v) for s, v in g23_traj if s <= grok_step]
        if at_grok_g23:
            _, g23_grok = at_grok_g23[-1]
        else:
            _, g23_grok = g23_traj[0]

        decline = g23_early / g23_grok if g23_grok > 1e-12 else float("inf")

        window = [(s, v) for s, v in g23_traj if early_step <= s <= grok_step]
        n_increases = sum(1 for i in range(1, len(window)) if window[i][1] > window[i-1][1])
        monotonic = n_increases <= MONOTONIC_TOL

        result.update({
            "g23_early": g23_early,
            "g23_early_step": early_step,
            "g23_grok": g23_grok,
            "decline": decline,
            "monotonic": monotonic,
            "n_increases": n_increases,
        })
    else:
        _, g23_peak = max(g23_traj, key=lambda x: x[1])
        _, g23_final = g23_traj[-1]
        result.update({
            "g23_early": g23_peak,
            "g23_grok": g23_final,
            "decline": g23_peak / g23_final if g23_final > 1e-12 else float("inf"),
            "monotonic": None,
        })

    # --- R analysis ---
    if grokked:
        pre_grok_R = [r for s, r in R_traj if s <= grok_step]
        R_early = np.mean(pre_grok_R) if pre_grok_R else None

        at_grok_R = [(s, r) for s, r in R_traj if s <= grok_step]
        R_grok = at_grok_R[-1][1] if at_grok_R else None
    else:
        R_early = np.mean([r for _, r in R_traj[:len(R_traj)//2]]) if R_traj else None
        R_grok = R_traj[-1][1] if R_traj else None

    kstar_mode = Counter(kstar_vals).most_common(1)[0][0]

    result.update({
        "R_early": R_early,
        "R_grok": R_grok,
        "kstar_mode": kstar_mode,
        "kstar_frac_1": sum(1 for k in kstar_vals if k == 1) / len(kstar_vals),
    })

    # store trajectories for plotting
    result["gram"] = gram
    result["metrics"] = metrics

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_dataset_results(dataset_name, all_results):
    """Generate summary plots for a dataset."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Thesis Table 7 — {dataset_name.upper()}  (Gram matrix, all layers flattened)",
                 fontsize=14, fontweight="bold")

    colors_wd1 = ["#1f77b4", "#2ca02c", "#d62728"]
    colors_wd0 = ["#aec7e8", "#98df8a", "#ff9896"]

    # --- Plot 1: g₂₃ = λ₂ − λ₃ from Gram matrix ---
    ax = axes[0, 0]
    for i, (label, res) in enumerate(all_results.items()):
        if res is None or "gram" not in res:
            continue
        gram = res["gram"]
        steps = [g["step"] for g in gram]
        g23 = [g["g23"] for g in gram]
        wd = res["wd"]
        color = colors_wd1[i % 3] if wd > 0 else colors_wd0[i % 3]
        ls = "-" if wd > 0 else "--"
        ax.plot(steps, g23, color=color, ls=ls, lw=1.2, label=label, alpha=0.8)
        if res.get("grok_step"):
            ax.axvline(res["grok_step"], color=color, ls=":", alpha=0.4)
    ax.set_xlabel("Step")
    ax.set_ylabel("g₂₃ = λ₂ − λ₃")
    ax.set_title("Gram eigenvalue gap g₂₃")
    ax.legend(fontsize=7, ncol=2)

    # --- Plot 2: g₂₃ zoomed near grok ---
    ax = axes[0, 1]
    for i, (label, res) in enumerate(all_results.items()):
        if res is None or "gram" not in res or not res.get("grokked"):
            continue
        gram = res["gram"]
        gs = res["grok_step"]
        window = [(g["step"], g["g23"]) for g in gram if g["step"] <= gs * 2]
        if not window:
            continue
        steps = [s for s, _ in window]
        vals = [v for _, v in window]
        color = colors_wd1[i % 3]
        ax.plot(steps, vals, color=color, lw=1.5, label=label, alpha=0.8)
        ax.axvline(gs, color=color, ls=":", alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("g₂₃")
    ax.set_title("g₂₃ near grokking (WD=1 only)")
    ax.legend(fontsize=7)

    # --- Plot 3: R trajectory ---
    ax = axes[0, 2]
    for i, (label, res) in enumerate(all_results.items()):
        if res is None or "gram" not in res:
            continue
        gram = res["gram"]
        steps = [g["step"] for g in gram]
        R_vals = [g["R"] for g in gram]
        wd = res["wd"]
        color = colors_wd1[i % 3] if wd > 0 else colors_wd0[i % 3]
        ls = "-" if wd > 0 else "--"
        ax.plot(steps, R_vals, color=color, ls=ls, lw=1.2, label=label, alpha=0.8)
        if res.get("grok_step"):
            ax.axvline(res["grok_step"], color=color, ls=":", alpha=0.4)
    ax.set_xlabel("Step")
    ax.set_ylabel("R (gap ratio)")
    ax.set_title(f"Gram gap ratio R")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)

    # --- Plot 4: k* trajectory ---
    ax = axes[1, 0]
    for i, (label, res) in enumerate(all_results.items()):
        if res is None or "gram" not in res:
            continue
        gram = res["gram"]
        steps = [g["step"] for g in gram]
        kstar = [g["kstar"] for g in gram]
        wd = res["wd"]
        color = colors_wd1[i % 3] if wd > 0 else colors_wd0[i % 3]
        ls = "-" if wd > 0 else "--"
        ax.plot(steps, kstar, color=color, ls=ls, lw=1.2, label=label, alpha=0.8, marker=".", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("k* (weighted)")
    ax.set_title("Weighted k*")
    ax.set_yticks(range(1, 11))
    ax.legend(fontsize=7, ncol=2)

    # --- Plot 5: Test accuracy ---
    ax = axes[1, 1]
    for i, (label, res) in enumerate(all_results.items()):
        if res is None or "metrics" not in res:
            continue
        met = res["metrics"]
        steps = [m["step"] for m in met]
        test_acc = [m["test_acc"] for m in met]
        wd = res["wd"]
        color = colors_wd1[i % 3] if wd > 0 else colors_wd0[i % 3]
        ls = "-" if wd > 0 else "--"
        ax.plot(steps, test_acc, color=color, ls=ls, lw=1.2, label=label, alpha=0.8)
    ax.axhline(GROK_THRESHOLD, color="gray", ls="--", alpha=0.5, label=f"threshold={GROK_THRESHOLD}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Test accuracy")
    ax.legend(fontsize=7, ncol=2)

    # --- Plot 6: Decline ratio bar chart ---
    ax = axes[1, 2]
    grok_runs = [(l, r) for l, r in all_results.items()
                 if r is not None and r.get("grokked") and r.get("decline") is not None]
    if grok_runs:
        labels_bar = [l for l, _ in grok_runs]
        declines = [r["decline"] for _, r in grok_runs]
        ax.bar(range(len(labels_bar)), declines, color="#1f77b4", alpha=0.7)
        ax.set_xticks(range(len(labels_bar)))
        ax.set_xticklabels(labels_bar, rotation=45, ha="right", fontsize=7)
        ax.axhline(1.0, color="red", ls="--", alpha=0.5)
        ax.set_ylabel("Decline ratio")
        ax.set_title("g₂₃ decline (grokking runs)")
    else:
        ax.text(0.5, 0.5, "No grokking runs", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("g₂₃ decline ratio")

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"thesis_table7_{dataset_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def print_table(dataset_name, all_results):
    """Print Table 7–style summary."""
    print(f"\n{'='*105}")
    print(f"  Table 7: {dataset_name.upper()}  —  Gram matrix (all layers, all W* flattened into δ_t)")
    print(f"{'='*105}")

    for label, res in all_results.items():
        if res is not None:
            print(f"  {res['n_layers']} layers, p = {res['p_total']:,} attn params, W = {res['gram_window']}")
            break

    header = f"{'Run':<28} {'Grok?':>5} {'Step':>7} {'g23_e':>12} {'g23_g':>12} {'Decl':>6} {'Mono':>5} {'R_e':>8} {'R_g':>8} {'k*':>3}"
    print(header)
    print("-" * len(header))

    grok_declines, grok_R_early, grok_R_grok, grok_kstar_1_frac = [], [], [], []

    for label, res in all_results.items():
        if res is None:
            print(f"{label:<28} {'SKIP':>5}")
            continue

        grok = "YES" if res["grokked"] else "no"
        step = str(res["grok_step"]) if res["grok_step"] else "-"
        g23_e = f"{res['g23_early']:.4e}" if res.get("g23_early") is not None else "-"
        g23_g = f"{res['g23_grok']:.4e}" if res.get("g23_grok") is not None else "-"
        decl = f"{res['decline']:.2f}" if res.get("decline") is not None and res["decline"] != float("inf") else "-"
        mono = "Y" if res.get("monotonic") is True else ("N" if res.get("monotonic") is False else "-")
        R_e = f"{res['R_early']:.2f}" if res.get("R_early") is not None else "-"
        R_g = f"{res['R_grok']:.2f}" if res.get("R_grok") is not None else "-"
        ks = str(res.get("kstar_mode", "-"))

        print(f"{label:<28} {grok:>5} {step:>7} {g23_e:>12} {g23_g:>12} {decl:>6} {mono:>5} {R_e:>8} {R_g:>8} {ks:>3}")

        if res["grokked"] and res.get("decline") is not None and res["decline"] != float("inf"):
            grok_declines.append(res["decline"])
        if res["grokked"] and res.get("R_early") is not None:
            grok_R_early.append(res["R_early"])
        if res["grokked"] and res.get("R_grok") is not None:
            grok_R_grok.append(res["R_grok"])
        if res["grokked"] and res.get("kstar_frac_1") is not None:
            grok_kstar_1_frac.append(res["kstar_frac_1"])

    print("-" * len(header))

    print(f"\n  Summary (grokking runs only):")
    if grok_declines:
        print(f"    g₂₃ decline ratio: mean={np.mean(grok_declines):.2f}×, "
              f"range=[{np.min(grok_declines):.2f}, {np.max(grok_declines):.2f}]")
    if grok_R_early:
        print(f"    R_early: mean={np.mean(grok_R_early):.2f}")
    if grok_R_grok:
        print(f"    R_grok:  mean={np.mean(grok_R_grok):.2f}")
    if grok_kstar_1_frac:
        n1 = sum(1 for f in grok_kstar_1_frac if f > 0.5)
        print(f"    k*=1 majority: {n1}/{len(grok_kstar_1_frac)} runs")

    grok_mono = [r for r in all_results.values() if r is not None and r.get("grokked")]
    if grok_mono:
        n_mono = sum(1 for r in grok_mono if r.get("monotonic") is True)
        print(f"    Monotonic g₂₃ decline: {n_mono}/{len(grok_mono)} runs")

    ctrl = [r for r in all_results.values() if r is not None and not r["grokked"]]
    if ctrl:
        print(f"    Control (WD=0) runs: {len(ctrl)}, none grokked (expected)")


def main():
    print("=" * 70)
    print("  Spectral Edge Thesis — Table 7 Replication (Dyck + SCAN)")
    print("  All quantities from Gram matrix of parameter updates")
    print("  δ_t = all W* from all layers, flattened")
    print("=" * 70)
    print(f"  MAX_FILE_MB = {MAX_FILE_MB}")
    print(f"  Gram window: Dyck W={GRAM_WINDOW_DYCK}, SCAN W={GRAM_WINDOW_SCAN}")
    print(f"  Grok threshold = {GROK_THRESHOLD}")
    print()

    all_dataset_results = {}

    for dataset_name, dcfg in DATASETS.items():
        print(f"\n{'─'*50}")
        print(f"  Dataset: {dataset_name.upper()}")
        print(f"{'─'*50}")

        results_dir = dcfg["results_dir"]
        if not results_dir.exists():
            print(f"  WARNING: {results_dir} not found, skipping")
            continue

        all_results = OrderedDict()

        dense_overrides = dcfg.get("dense_overrides", {})

        for wd in WD_VALUES:
            for seed in SEEDS:
                wd_str = f"{wd:.1f}" if wd == int(wd) else str(wd)

                # Use dense checkpoint if available
                if (wd, seed) in dense_overrides:
                    fname = dense_overrides[(wd, seed)]
                    label = f"wd={wd_str} s={seed} (dense)"
                else:
                    fname = dcfg["file_pattern"].format(wd=wd_str, seed=seed)
                    label = f"wd={wd_str} s={seed}"

                fpath = results_dir / fname

                if not fpath.exists():
                    print(f"  WARNING: {fpath} not found")
                    all_results[label] = None
                    continue

                data = load_run(fpath)
                if data is None:
                    all_results[label] = None
                    continue

                print(f"    Analyzing {label}...")
                res = analyze_run(data, dcfg)
                all_results[label] = res

                del data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print_table(dataset_name, all_results)
        plot_dataset_results(dataset_name, all_results)

        save_results = {}
        for label, res in all_results.items():
            if res is None:
                save_results[label] = None
                continue
            save_res = {k: v for k, v in res.items() if k not in ("gram", "metrics")}
            save_results[label] = save_res

        out_pt = RESULTS_DIR / f"thesis_table7_{dataset_name}.pt"
        torch.save(save_results, out_pt)
        print(f"  Results saved: {out_pt}")

        all_dataset_results[dataset_name] = all_results

    # Cross-dataset summary
    print(f"\n\n{'='*70}")
    print("  CROSS-DATASET SUMMARY")
    print(f"{'='*70}")
    for ds_name, ds_results in all_dataset_results.items():
        grok_runs = [r for r in ds_results.values() if r is not None and r.get("grokked")]
        ctrl_runs = [r for r in ds_results.values() if r is not None and not r["grokked"]]
        declines = [r["decline"] for r in grok_runs if r.get("decline") is not None and r["decline"] != float("inf")]
        print(f"\n  {ds_name.upper()}:")
        print(f"    Grokking runs: {len(grok_runs)}/6")
        print(f"    Control runs (no grok): {len(ctrl_runs)}/6")
        if declines:
            print(f"    Mean g₂₃ decline: {np.mean(declines):.2f}×")
        mono_runs = [r for r in grok_runs if r.get("monotonic") is True]
        print(f"    Monotonic decline: {len(mono_runs)}/{len(grok_runs)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
