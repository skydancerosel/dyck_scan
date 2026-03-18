#!/usr/bin/env python3
"""
Spectral geometry analysis for Dyck grokking.

Tests the conjecture: grokking is mediated by spectral symmetry-breaking
in attention weight matrices.

Pipeline:
  1. SVD of W_Q, W_K at each checkpoint -> spectral gaps g12, g23
  2. Matrix commutator ||[W_Q, W_K]||_F at each checkpoint
  3. Load SGD commutator defect from pre-computed results
  4. Per-head analysis (d_head blocks)
  5. Temporal ordering of key events
  6. Phase portraits (spectral gap x non-commutativity)

Figures produced:
  SVD1  -- 5-panel timeseries (g12, g23, comm, SGD defect, acc)
  SVD2  -- Scatter: SVD gaps vs both commutators
  SVD3  -- Phase-colored scatter (pre/trans/post grok)
  SVD4  -- Per-head SVD gap vs per-head comm
  SVD5  -- Narrative test (all quantities normalized [0,1])
  SVD6  -- Grok (wd=1) vs control (wd=0) SVD dynamics
  PP1   -- Hero phase portrait (wd=1.0, seed=42)
  PP2   -- Grid: 2 wd × 3 seeds phase portraits
  PP3   -- Grok vs memorizing in same phase space
  PP4   -- 3D portrait (g12, g23, comm)
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

SWEEP_DIR = Path(__file__).parent / "../dyck_sweep_results"
COMM_DIR  = Path(__file__).parent / "figures"
OUT_DIR   = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = [42, 137, 2024]
WDS   = [1.0, 0.0]
LAYER_IDX = 0       # layer to analyze (layer 0 for Dyck)
N_HEADS = 4
D_MODEL = 128
D_HEAD  = D_MODEL // N_HEADS  # 32
SVD_TOPK = 5        # track top-5 singular values
SMOOTH_WINDOW = 3   # rolling mean window for phase portraits

# ═══════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_run(wd, seed):
    """Load a single training run."""
    path = SWEEP_DIR / f"dyck_wd{wd}_s{seed}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def load_sgd_defect():
    """Load SGD commutator defect from generalization dynamics results.

    Uses generalization_dynamics_results.pt (100-step resolution) rather than
    commutator_results.pt (200-step resolution) because the finer resolution
    is needed for accurate spike detection — the defect can explode between
    step 0 and 200, inflating the baseline if the first spike is missed.
    """
    path = COMM_DIR / "dyck_generalization_dynamics_results.pt"
    if not path.exists():
        # Fallback to commutator results
        path = COMM_DIR / "dyck_commutator_results.pt"
        if not path.exists():
            return {}
        return {"_source": "commutator", "_data": torch.load(path, map_location="cpu", weights_only=False)}
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {"_source": "gen_dynamics", "_data": data.get("all_runs", {})}


# ═══════════════════════════════════════════════════════════════════════════
# Core computations
# ═══════════════════════════════════════════════════════════════════════════

def compute_spectral_quantities(attn_logs, layer_idx=0):
    """
    For each checkpoint, compute:
      - Singular values of WQ, WK (top-k)
      - Spectral gaps g12, g23 for WQ
      - ||[WQ, WK]||_F (matrix commutator Frobenius norm)
    """
    steps = []
    sv_q_all = []   # [n_checkpoints, topk]
    sv_k_all = []
    g12_q = []      # sigma1 - sigma2 for WQ
    g23_q = []      # sigma2 - sigma3 for WQ
    g12_k = []
    g23_k = []
    comm_norms = [] # ||[WQ, WK]||_F

    for snap in attn_logs:
        step = snap["step"]
        layer = snap["layers"][layer_idx]
        WQ = layer["WQ"].numpy().astype(np.float64)
        WK = layer["WK"].numpy().astype(np.float64)

        # SVD (singular values only)
        sq = np.linalg.svd(WQ, compute_uv=False)[:SVD_TOPK]
        sk = np.linalg.svd(WK, compute_uv=False)[:SVD_TOPK]

        # Spectral gaps
        g12_q.append(sq[0] - sq[1])
        g23_q.append(sq[1] - sq[2])
        g12_k.append(sk[0] - sk[1])
        g23_k.append(sk[1] - sk[2])

        # Matrix commutator
        comm = WQ @ WK - WK @ WQ
        comm_norm = np.linalg.norm(comm, "fro")

        steps.append(step)
        sv_q_all.append(sq)
        sv_k_all.append(sk)
        comm_norms.append(comm_norm)

    return {
        "steps": np.array(steps),
        "sv_q": np.array(sv_q_all),
        "sv_k": np.array(sv_k_all),
        "g12_q": np.array(g12_q),
        "g23_q": np.array(g23_q),
        "g12_k": np.array(g12_k),
        "g23_k": np.array(g23_k),
        "comm_norms": np.array(comm_norms),
    }


def compute_per_head(attn_logs, layer_idx=0):
    """
    Per-head SVD gap and commutator for each checkpoint.
    Returns [n_checkpoints, n_heads] arrays.
    """
    steps = []
    head_gaps = []
    head_comms = []

    for snap in attn_logs:
        step = snap["step"]
        layer = snap["layers"][layer_idx]
        WQ = layer["WQ"].numpy().astype(np.float64)
        WK = layer["WK"].numpy().astype(np.float64)

        hg = []
        hc = []
        for h in range(N_HEADS):
            s, e = h * D_HEAD, (h + 1) * D_HEAD
            q_block = WQ[s:e, s:e]
            k_block = WK[s:e, s:e]

            sq = np.linalg.svd(q_block, compute_uv=False)
            gap = sq[0] - sq[1]
            comm = np.linalg.norm(q_block @ k_block - k_block @ q_block, "fro")
            hg.append(gap)
            hc.append(comm)

        steps.append(step)
        head_gaps.append(hg)
        head_comms.append(hc)

    return {
        "steps": np.array(steps),
        "head_gaps": np.array(head_gaps),
        "head_comms": np.array(head_comms),
    }


def compute_pca_and_rotation(attn_logs, layer_idx=0, top_k=5):
    """Expanding-window PCA on QK update deltas + PC rotation angles.
    (from grok_geometry_conjecture_test.py reference)"""
    deltas, delta_steps = [], []
    for i in range(1, len(attn_logs)):
        WQ0 = attn_logs[i-1]["layers"][layer_idx]["WQ"].float().numpy().flatten()
        WK0 = attn_logs[i-1]["layers"][layer_idx]["WK"].float().numpy().flatten()
        WQ1 = attn_logs[i]["layers"][layer_idx]["WQ"].float().numpy().flatten()
        WK1 = attn_logs[i]["layers"][layer_idx]["WK"].float().numpy().flatten()
        deltas.append(np.concatenate([WQ1 - WQ0, WK1 - WK0]))
        delta_steps.append(attn_logs[i]["step"])

    pca_steps, explained_list = [], []
    rot_steps, rot_list = [], []
    prev_Vt = None

    for t in range(3, len(deltas) + 1):
        step = delta_steps[t - 1]
        X = np.stack(deltas[:t])
        X -= X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        eigvals = (S ** 2) / max(X.shape[0] - 1, 1)
        total = eigvals.sum()
        if total < 1e-30:
            continue

        k = min(top_k, len(eigvals))
        ratios = np.zeros(top_k)
        ratios[:k] = eigvals[:k] / total
        pca_steps.append(step)
        explained_list.append(ratios)

        if prev_Vt is not None:
            k_rot = min(3, Vt.shape[0], prev_Vt.shape[0])
            thetas = []
            for i in range(k_rot):
                dot = np.clip(np.abs(np.dot(Vt[i], prev_Vt[i])), 0, 1)
                thetas.append(np.arccos(dot))
            rot_steps.append(step)
            rot_list.append(thetas)

        prev_Vt = Vt[:min(3, Vt.shape[0])].copy()

    return (np.array(pca_steps), np.array(explained_list),
            np.array(rot_steps), np.array(rot_list) if rot_list else np.empty((0, 3)))


def get_metrics_at_steps(metrics, target_steps):
    """Interpolate metrics to match checkpoint steps."""
    m_steps = np.array([m["step"] for m in metrics])
    m_train_acc = np.array([m["train_acc"] for m in metrics])
    m_test_acc = np.array([m["test_acc"] for m in metrics])

    train_acc = np.interp(target_steps, m_steps, m_train_acc)
    test_acc = np.interp(target_steps, m_steps, m_test_acc)
    return train_acc, test_acc


def _get_sgd_raw(sgd_data, wd, seed):
    """Get raw SGD defect (steps, values) at native resolution."""
    source = sgd_data.get("_source", "")
    data = sgd_data.get("_data", sgd_data)

    if source == "gen_dynamics":
        wd_str = f"wd{wd}" if not str(wd).startswith("wd") else str(wd)
        key = (wd_str, seed)
        if key not in data:
            return None, None
        recs = data[key].get("records", [])
        if not recs:
            return None, None
        return (np.array([r["step"] for r in recs]),
                np.array([r["defect_median"] for r in recs]))
    else:
        key = (wd, seed)
        if key not in data:
            return None, None
        dt = data[key].get("defect_trace", [])
        if not dt:
            return None, None
        return (np.array([d["step"] for d in dt]),
                np.array([d["defect"] for d in dt]))


def get_sgd_defect_at_steps(sgd_data, wd, seed, target_steps):
    """Get SGD defect interpolated to checkpoint steps."""
    d_steps, d_vals = _get_sgd_raw(sgd_data, wd, seed)
    if d_steps is None:
        return np.zeros_like(target_steps, dtype=float)
    return np.interp(target_steps, d_steps, d_vals)


def find_sgd_spike_step(sgd_data, wd, seed, threshold_factor=10, min_defect=20):
    """Detect SGD defect spike at native resolution (not interpolated).

    This avoids the baseline inflation that occurs when coarse checkpoint
    intervals miss the early ramp-up of the defect.
    """
    d_steps, d_vals = _get_sgd_raw(sgd_data, wd, seed)
    if d_steps is None or len(d_vals) < 3:
        return None
    baseline = np.median(d_vals[:3])
    baseline = max(baseline, 0.1)
    for i in range(2, len(d_vals)):
        if d_vals[i] > threshold_factor * baseline and d_vals[i] > min_defect:
            return int(d_steps[i])
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def norm01(x):
    mn, mx = x.min(), x.max()
    r = mx - mn
    if r < 1e-30:
        return np.zeros_like(x)
    return (x - mn) / r


def rolling_mean(x, w=3):
    if len(x) < w:
        return x.copy()
    kernel = np.ones(w) / w
    pad_l = w // 2
    pad_r = w - pad_l - 1
    padded = np.concatenate([np.repeat(x[:1], pad_l), x, np.repeat(x[-1:], pad_r)])
    return np.convolve(padded, kernel, mode="valid")[:len(x)]


def find_grok_step(metrics, threshold=0.9, key="test_acc"):
    """Find first step where test accuracy >= threshold."""
    for m in metrics:
        if m[key] >= threshold:
            return m["step"]
    return None


def find_events(spec, sgd_defect, metrics, grok_step, sgd_spike_step=None):
    """Identify key temporal events."""
    steps = spec["steps"]
    g12 = spec["g12_q"]
    g23 = spec["g23_q"]
    comm = spec["comm_norms"]

    events = {}

    # g23 decline: peak in first ~15 checkpoints
    early = min(15, len(g23))
    events["g23_peak"] = steps[np.argmax(g23[:early])]

    # sigma1 ≈ sigma2 minimum (max near-degeneracy)
    events["g12_min"] = steps[np.argmin(g12)]

    # SGD spike: use native-resolution detection if provided
    if sgd_spike_step is not None:
        events["sgd_spike"] = sgd_spike_step
    elif len(sgd_defect) >= 3:
        # Fallback: detect from interpolated data using step-0 baseline
        baseline = max(sgd_defect[0], 0.1)
        for i in range(1, len(sgd_defect)):
            d = sgd_defect[i]
            if d > 10 * baseline and d > 20:
                events["sgd_spike"] = steps[min(i, len(steps) - 1)]
                break

    # Commutator peak (skip first 3 for stability)
    skip = min(3, len(comm) - 1)
    events["comm_peak"] = steps[skip + np.argmax(comm[skip:])]

    # g12 maximum (one mode dominates)
    events["g12_max"] = steps[skip + np.argmax(g12[skip:])]

    # Commutator collapse: first step after peak where comm < 50% of peak
    peak_idx = skip + np.argmax(comm[skip:])
    peak_val = comm[peak_idx]
    for i in range(peak_idx + 1, len(comm)):
        if comm[i] < 0.5 * peak_val:
            events["comm_collapse"] = steps[i]
            break

    if grok_step is not None:
        events["grok"] = grok_step

    return events


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def fig_SVD1(all_specs, all_sgd, all_metrics, all_grok_steps):
    """5-panel timeseries for each (wd, seed) condition."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    titles = ["σ₁−σ₂ (W_Q)", "σ₂−σ₃ (W_Q)", "‖[W_Q,W_K]‖_F",
              "SGD Defect", "Test Accuracy"]

    colors_wd1 = ["#e74c3c", "#c0392b", "#a93226"]
    colors_wd0 = ["#3498db", "#2980b9", "#2471a3"]

    for idx, (wd, seed) in enumerate([(w, s) for w in WDS for s in SEEDS]):
        key = (wd, seed)
        if key not in all_specs:
            continue
        spec = all_specs[key]
        steps = spec["steps"]
        color = colors_wd1[SEEDS.index(seed)] if wd == 1.0 else colors_wd0[SEEDS.index(seed)]
        ls = "-" if wd == 1.0 else "--"
        label = f"wd={wd}, s={seed}"
        alpha = 0.8

        axes[0].plot(steps, spec["g12_q"], ls, color=color, label=label, alpha=alpha)
        axes[1].plot(steps, spec["g23_q"], ls, color=color, alpha=alpha)
        axes[2].plot(steps, spec["comm_norms"], ls, color=color, alpha=alpha)

        sgd = all_sgd.get(key, np.zeros_like(steps))
        axes[3].plot(steps, sgd, ls, color=color, alpha=alpha)

        _, test_acc = get_metrics_at_steps(all_metrics[key], steps)
        axes[4].plot(steps, test_acc, ls, color=color, alpha=alpha)

        # Mark grok step
        gs = all_grok_steps.get(key)
        if gs is not None:
            for ax in axes:
                ax.axvline(gs, color=color, alpha=0.3, ls=":")

    for ax, title in zip(axes, titles):
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Training step")
    fig.suptitle("Dyck: Spectral Quantities Over Training", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figSVD1_dyck_timeseries.png", dpi=150)
    plt.close(fig)
    print("  saved figSVD1")


def fig_SVD2(all_specs, all_sgd):
    """Scatter: SVD gaps vs both commutators."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for wd, seed in [(w, s) for w in WDS for s in SEEDS]:
        key = (wd, seed)
        if key not in all_specs:
            continue
        spec = all_specs[key]
        marker = "o" if wd == 1.0 else "x"
        color = "#e74c3c" if wd == 1.0 else "#3498db"
        alpha = 0.4
        label = f"wd={wd}, s={seed}"

        # g12 vs matrix commutator
        axes[0].scatter(spec["g12_q"], spec["comm_norms"], marker=marker,
                       c=color, alpha=alpha, s=15, label=label)
        # g12 vs SGD defect
        sgd = all_sgd.get(key, np.zeros_like(spec["steps"]))
        axes[1].scatter(spec["g12_q"], sgd, marker=marker,
                       c=color, alpha=alpha, s=15)

    axes[0].set_xlabel("σ₁−σ₂ (W_Q)")
    axes[0].set_ylabel("‖[W_Q,W_K]‖_F")
    axes[0].set_title("SVD Gap vs Matrix Commutator")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("σ₁−σ₂ (W_Q)")
    axes[1].set_ylabel("SGD Defect")
    axes[1].set_title("SVD Gap vs SGD Defect")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Dyck: Cross-sectional Correlations", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figSVD2_dyck_scatter.png", dpi=150)
    plt.close(fig)
    print("  saved figSVD2")


def fig_SVD3(all_specs, all_metrics, all_grok_steps):
    """Phase-colored scatter (pre/trans/post grok)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    phase_colors = {"pre": "#3498db", "trans": "#f39c12", "post": "#2ecc71"}

    for wd, seed in [(w, s) for w in [1.0] for s in SEEDS]:
        key = (wd, seed)
        if key not in all_specs:
            continue
        spec = all_specs[key]
        steps = spec["steps"]
        _, test_acc = get_metrics_at_steps(all_metrics[key], steps)
        gs = all_grok_steps.get(key)

        # Classify phases
        for i, step in enumerate(steps):
            if test_acc[i] < 0.5:
                phase = "pre"
            elif test_acc[i] < 0.9:
                phase = "trans"
            else:
                phase = "post"
            ax.scatter(spec["g12_q"][i], spec["comm_norms"][i],
                      c=phase_colors[phase], alpha=0.5, s=20,
                      marker="o" if seed == 42 else ("s" if seed == 137 else "^"))

    # Legend
    for phase, color in phase_colors.items():
        ax.scatter([], [], c=color, label=phase, s=40)
    ax.legend()
    ax.set_xlabel("σ₁−σ₂ (W_Q)")
    ax.set_ylabel("‖[W_Q,W_K]‖_F")
    ax.set_title("Dyck: Phase-Colored Scatter (wd=1.0)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figSVD3_dyck_phase_scatter.png", dpi=150)
    plt.close(fig)
    print("  saved figSVD3")


def fig_SVD4(all_head_data, all_grok_steps):
    """Per-head SVD gap vs per-head commutator."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    head_colors = plt.cm.Set1(np.linspace(0, 1, N_HEADS))

    # wd=1.0 seeds
    for si, seed in enumerate(SEEDS):
        key = (1.0, seed)
        if key not in all_head_data:
            continue
        hd = all_head_data[key]
        for h in range(N_HEADS):
            axes[0].scatter(hd["head_gaps"][:, h], hd["head_comms"][:, h],
                          c=[head_colors[h]], alpha=0.3, s=10,
                          label=f"head {h}" if si == 0 else None)

    # wd=0.0 seeds
    for si, seed in enumerate(SEEDS):
        key = (0.0, seed)
        if key not in all_head_data:
            continue
        hd = all_head_data[key]
        for h in range(N_HEADS):
            axes[1].scatter(hd["head_gaps"][:, h], hd["head_comms"][:, h],
                          c=[head_colors[h]], alpha=0.3, s=10,
                          label=f"head {h}" if si == 0 else None)

    for ax, title in zip(axes, ["wd=1.0 (grokking)", "wd=0.0 (memorizing)"]):
        ax.set_xlabel("Per-head σ₁−σ₂")
        ax.set_ylabel("Per-head ‖[Q,K]‖_F")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Dyck: Per-Head Spectral Gap vs Commutator", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figSVD4_dyck_per_head.png", dpi=150)
    plt.close(fig)
    print("  saved figSVD4")


def fig_SVD5(all_specs, all_sgd, all_metrics, all_grok_steps):
    """Narrative test: all quantities normalized [0,1] overlaid."""
    n_grok = sum(1 for k in all_grok_steps if k[0] == 1.0 and all_grok_steps[k] is not None)
    if n_grok == 0:
        print("  skipping figSVD5 (no grokking runs)")
        return

    fig, axes = plt.subplots(min(n_grok, 3), 1, figsize=(14, 4 * min(n_grok, 3)),
                              squeeze=False)
    ax_idx = 0

    for seed in SEEDS:
        key = (1.0, seed)
        if key not in all_specs or all_grok_steps.get(key) is None:
            continue
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx, 0]
        spec = all_specs[key]
        steps = spec["steps"]
        sgd = all_sgd.get(key, np.zeros_like(steps))
        _, test_acc = get_metrics_at_steps(all_metrics[key], steps)
        gs = all_grok_steps[key]

        # Normalize
        ax.plot(steps, norm01(spec["g12_q"]), "-", color="#e74c3c", label="σ₁−σ₂", lw=2)
        ax.plot(steps, norm01(spec["g23_q"]), "-", color="#f39c12", label="σ₂−σ₃", lw=1.5)
        ax.plot(steps, norm01(spec["comm_norms"]), "--", color="#9b59b6",
                label="‖[Q,K]‖_F", lw=2)
        if sgd.max() > 0:
            log_sgd = np.log10(np.maximum(sgd, 1e-10))
            ax.plot(steps, norm01(log_sgd), "-.", color="#2ecc71",
                    label="log₁₀(SGD)", lw=1.5)
        ax.plot(steps, test_acc, ":", color="#00bcd4", label="Test Acc", lw=2)
        ax.axvline(gs, color="gray", ls="--", alpha=0.5, label=f"Grok ({gs})")
        ax.set_ylabel("Normalized [0,1]")
        ax.set_title(f"seed={seed}")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    axes[-1, 0].set_xlabel("Training step")
    fig.suptitle("Dyck: Narrative Test (wd=1.0)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figSVD5_dyck_narrative.png", dpi=150)
    plt.close(fig)
    print("  saved figSVD5")


def fig_SVD6(all_specs, all_metrics, all_grok_steps):
    """Grok (wd=1) vs control (wd=0) SVD dynamics."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    titles = ["σ₁−σ₂ (W_Q)", "‖[W_Q,W_K]‖_F", "Test Accuracy"]

    for wd in WDS:
        for seed in SEEDS:
            key = (wd, seed)
            if key not in all_specs:
                continue
            spec = all_specs[key]
            steps = spec["steps"]
            _, test_acc = get_metrics_at_steps(all_metrics[key], steps)

            color = "#e74c3c" if wd == 1.0 else "#3498db"
            ls = "-" if wd == 1.0 else "--"
            alpha = 0.7
            label = f"wd={wd}, s={seed}"

            axes[0].plot(steps, spec["g12_q"], ls, color=color, alpha=alpha, label=label)
            axes[1].plot(steps, spec["comm_norms"], ls, color=color, alpha=alpha)
            axes[2].plot(steps, test_acc, ls, color=color, alpha=alpha)

    for ax, title in zip(axes, titles):
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Training step")
    fig.suptitle("Dyck: Grok (wd=1.0) vs Control (wd=0.0)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figSVD6_dyck_grok_vs_control.png", dpi=150)
    plt.close(fig)
    print("  saved figSVD6")


def _phase_portrait_ax(ax, spec, sgd_defect, metrics, grok_step,
                        color_by="step", title="", show_events=True,
                        sgd_spike_step=None):
    """Draw a phase portrait on an axis."""
    steps = spec["steps"]
    g12 = spec["g12_q"]
    comm = spec["comm_norms"]

    # Smooth
    g12_s = rolling_mean(g12, SMOOTH_WINDOW)
    comm_s = rolling_mean(comm, SMOOTH_WINDOW)

    # Raw as faint ghost
    ax.plot(g12, comm, color="gray", alpha=0.15, lw=0.5, zorder=1)

    # Color trajectory
    if color_by == "step":
        cmap = plt.cm.viridis
        norm = Normalize(vmin=steps.min(), vmax=steps.max())
        colors = cmap(norm(steps))
        label_str = "Training step"
    else:  # test_acc
        _, test_acc = get_metrics_at_steps(metrics, steps)
        cmap = plt.cm.RdYlGn
        norm = Normalize(vmin=0, vmax=1)
        colors = cmap(norm(test_acc))
        label_str = "Test accuracy"

    # Plot colored segments
    for i in range(len(steps) - 1):
        ax.plot(g12_s[i:i+2], comm_s[i:i+2], color=colors[i], lw=2, zorder=2)

    # Direction arrows
    arrow_every = max(1, len(steps) // 15)
    for i in range(0, len(steps) - 1, arrow_every):
        dx = g12_s[min(i+1, len(g12_s)-1)] - g12_s[i]
        dy = comm_s[min(i+1, len(comm_s)-1)] - comm_s[i]
        if abs(dx) > 1e-10 or abs(dy) > 1e-10:
            ax.annotate("", xy=(g12_s[i] + dx * 0.5, comm_s[i] + dy * 0.5),
                        xytext=(g12_s[i], comm_s[i]),
                        arrowprops=dict(arrowstyle="->", color=colors[i],
                                       lw=1.5), zorder=3)

    # Event markers
    if show_events:
        events = find_events(spec, sgd_defect, metrics, grok_step, sgd_spike_step)
        # Init
        ax.scatter(g12_s[0], comm_s[0], c="gray", marker="o", s=80,
                  zorder=5, edgecolors="black", label="Init")

        # SGD spike
        if "sgd_spike" in events:
            idx = np.argmin(np.abs(steps - events["sgd_spike"]))
            ax.scatter(g12_s[idx], comm_s[idx], c="green", marker="^", s=80,
                      zorder=5, edgecolors="black", label="SGD spike")

        # Commutator peak
        if "comm_peak" in events:
            idx = np.argmin(np.abs(steps - events["comm_peak"]))
            ax.scatter(g12_s[idx], comm_s[idx], c="red", marker="D", s=80,
                      zorder=5, edgecolors="black", label="Comm peak")

        # g12 minimum
        if "g12_min" in events:
            idx = np.argmin(np.abs(steps - events["g12_min"]))
            ax.scatter(g12_s[idx], comm_s[idx], c="purple", marker="v", s=80,
                      zorder=5, edgecolors="black", label="σ₁≈σ₂")

        # Grok
        if grok_step is not None:
            idx = np.argmin(np.abs(steps - grok_step))
            if idx < len(g12_s):
                ax.scatter(g12_s[idx], comm_s[idx], c="orange", marker="*", s=200,
                          zorder=6, edgecolors="black", label="GROK")

    # Phase region shading
    all_g12 = g12_s
    all_comm = comm_s
    gap_thresh = np.percentile(all_g12, 40)
    comm_thresh = np.percentile(all_comm, 60)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Competition (left strip)
    ax.axvspan(xlim[0], gap_thresh, alpha=0.045, color="purple", zorder=0)
    # Instability (top-right)
    ax.fill_between([gap_thresh, xlim[1]], comm_thresh, ylim[1],
                    alpha=0.045, color="red", zorder=0)
    # Alignment (bottom-right)
    ax.fill_between([gap_thresh, xlim[1]], ylim[0], comm_thresh,
                    alpha=0.045, color="green", zorder=0)
    # Dashed boundaries
    ax.axvline(gap_thresh, ls="--", color="gray", alpha=0.3, lw=0.8)
    ax.axhline(comm_thresh, ls="--", color="gray", alpha=0.3, lw=0.8)

    ax.set_xlabel("σ₁ − σ₂ (spectral gap)")
    ax.set_ylabel("‖[W_Q, W_K]‖_F (non-commutativity)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=label_str, fraction=0.03, pad=0.02)


def fig_PP1(all_specs, all_sgd, all_metrics, all_grok_steps, all_sgd_spikes=None):
    """Hero phase portrait (wd=1.0, seed=42)."""
    key = (1.0, 42)
    if key not in all_specs:
        print("  skipping figPP1 (no wd=1.0, s=42)")
        return
    spike = (all_sgd_spikes or {}).get(key)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    _phase_portrait_ax(axes[0], all_specs[key], all_sgd.get(key, np.zeros(1)),
                       all_metrics[key], all_grok_steps.get(key),
                       color_by="step", title="Colored by training step",
                       sgd_spike_step=spike)
    axes[0].legend(fontsize=7, loc="upper left")

    _phase_portrait_ax(axes[1], all_specs[key], all_sgd.get(key, np.zeros(1)),
                       all_metrics[key], all_grok_steps.get(key),
                       color_by="test_acc", title="Colored by test accuracy",
                       show_events=False)

    fig.suptitle("Dyck: Hero Phase Portrait (wd=1.0, seed=42, layer 0)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPP1_dyck_hero.png", dpi=150)
    plt.close(fig)
    print("  saved figPP1")


def fig_PP2(all_specs, all_sgd, all_metrics, all_grok_steps, all_sgd_spikes=None):
    """Grid: 2 wd × 3 seeds phase portraits."""
    fig, axes = plt.subplots(len(WDS), len(SEEDS), figsize=(18, 12))

    for i, wd in enumerate(WDS):
        for j, seed in enumerate(SEEDS):
            key = (wd, seed)
            ax = axes[i, j]
            if key not in all_specs:
                ax.set_visible(False)
                continue
            sgd = all_sgd.get(key, np.zeros_like(all_specs[key]["steps"]))
            spike = (all_sgd_spikes or {}).get(key)
            _phase_portrait_ax(ax, all_specs[key], sgd,
                              all_metrics[key], all_grok_steps.get(key),
                              color_by="step",
                              title=f"wd={wd}, seed={seed}",
                              sgd_spike_step=spike)
            if i == 0 and j == 0:
                ax.legend(fontsize=6, loc="upper left")

    fig.suptitle("Dyck: Phase Portraits Grid", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPP2_dyck_grid.png", dpi=150)
    plt.close(fig)
    print("  saved figPP2")


def fig_PP3(all_specs, all_sgd, all_metrics, all_grok_steps):
    """Grok vs memorizing in same phase space."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot wd=1.0 (grokking) in warm colors
    for si, seed in enumerate(SEEDS):
        key = (1.0, seed)
        if key not in all_specs:
            continue
        spec = all_specs[key]
        g12_s = rolling_mean(spec["g12_q"], SMOOTH_WINDOW)
        comm_s = rolling_mean(spec["comm_norms"], SMOOTH_WINDOW)
        color = ["#e74c3c", "#c0392b", "#a93226"][si]
        ax.plot(g12_s, comm_s, "-", color=color, lw=2, alpha=0.8,
                label=f"wd=1.0, s={seed} (grok)")
        # Mark start and end
        ax.scatter(g12_s[0], comm_s[0], c="gray", marker="o", s=60, zorder=5)
        gs = all_grok_steps.get(key)
        if gs is not None:
            idx = np.argmin(np.abs(spec["steps"] - gs))
            if idx < len(g12_s):
                ax.scatter(g12_s[idx], comm_s[idx], c="orange", marker="*",
                          s=150, zorder=6, edgecolors="black")

    # Plot wd=0.0 (memorizing) in cool colors
    for si, seed in enumerate(SEEDS):
        key = (0.0, seed)
        if key not in all_specs:
            continue
        spec = all_specs[key]
        g12_s = rolling_mean(spec["g12_q"], SMOOTH_WINDOW)
        comm_s = rolling_mean(spec["comm_norms"], SMOOTH_WINDOW)
        color = ["#3498db", "#2980b9", "#2471a3"][si]
        ax.plot(g12_s, comm_s, "--", color=color, lw=2, alpha=0.8,
                label=f"wd=0.0, s={seed} (memo)")
        ax.scatter(g12_s[0], comm_s[0], c="gray", marker="o", s=60, zorder=5)
        ax.scatter(g12_s[-1], comm_s[-1], c=color, marker="s", s=60, zorder=5)

    ax.set_xlabel("σ₁ − σ₂ (spectral gap)")
    ax.set_ylabel("‖[W_Q, W_K]‖_F (non-commutativity)")
    ax.set_title("Dyck: Grokking vs Memorizing Phase Trajectories")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPP3_dyck_grok_vs_memo.png", dpi=150)
    plt.close(fig)
    print("  saved figPP3")


def fig_PP4(all_specs, all_grok_steps):
    """3D phase portrait (g12, g23, comm)."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    for wd in WDS:
        for si, seed in enumerate(SEEDS):
            key = (wd, seed)
            if key not in all_specs:
                continue
            spec = all_specs[key]
            g12 = rolling_mean(spec["g12_q"], SMOOTH_WINDOW)
            g23 = rolling_mean(spec["g23_q"], SMOOTH_WINDOW)
            comm = rolling_mean(spec["comm_norms"], SMOOTH_WINDOW)
            steps = spec["steps"]

            color = "#e74c3c" if wd == 1.0 else "#3498db"
            ls = "-" if wd == 1.0 else "--"
            label = f"wd={wd}, s={seed}"
            ax.plot(g12, g23, comm, ls, color=color, alpha=0.7, label=label, lw=1.5)

            # Mark start
            ax.scatter(g12[0], g23[0], comm[0], c="gray", marker="o", s=50)
            # Mark grok
            gs = all_grok_steps.get(key)
            if gs is not None:
                idx = np.argmin(np.abs(steps - gs))
                if idx < len(g12):
                    ax.scatter(g12[idx], g23[idx], comm[idx], c="orange",
                              marker="*", s=150, edgecolors="black")

    ax.set_xlabel("σ₁ − σ₂")
    ax.set_ylabel("σ₂ − σ₃")
    ax.set_zlabel("‖[W_Q, W_K]‖_F")
    ax.set_title("Dyck: 3D Phase Portrait")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPP4_dyck_3d.png", dpi=150)
    plt.close(fig)
    print("  saved figPP4")


# ═══════════════════════════════════════════════════════════════════════════
# Temporal ordering report
# ═══════════════════════════════════════════════════════════════════════════

def print_temporal_ordering(all_specs, all_sgd, all_metrics, all_grok_steps,
                            all_sgd_spikes=None):
    """Print temporal ordering of key events for each grokking run."""
    predicted = ["g23_peak", "g12_min", "sgd_spike", "comm_peak", "g12_max", "grok"]
    predicted_labels = ["g₂₃↓", "σ₁≈σ₂", "D_SGD↑", "‖[Q,K]‖peak", "σ₁≫σ₂", "grok"]

    print("\n" + "=" * 70)
    print("TEMPORAL ORDERING OF KEY EVENTS")
    print("Predicted: " + " → ".join(predicted_labels))
    print("=" * 70)

    for seed in SEEDS:
        key = (1.0, seed)
        if key not in all_specs:
            continue
        spec = all_specs[key]
        sgd = all_sgd.get(key, np.zeros_like(spec["steps"]))
        gs = all_grok_steps.get(key)
        spike = (all_sgd_spikes or {}).get(key)

        events = find_events(spec, sgd, all_metrics[key], gs, spike)
        print(f"\n  wd=1.0, seed={seed}:")
        # Sort by step
        sorted_events = sorted(events.items(), key=lambda x: x[1])
        for name, step in sorted_events:
            label = predicted_labels[predicted.index(name)] if name in predicted else name
            print(f"    {label:>12s}: step {step:>6d}")

        # Check ordering
        actual_order = [name for name, _ in sorted_events if name in predicted]
        matches = sum(1 for a, b in zip(actual_order, predicted) if a == b)
        print(f"    Order match: {matches}/{len(predicted)} positions correct")
        print(f"    Actual: {' → '.join(predicted_labels[predicted.index(n)] for n in actual_order if n in predicted)}")


# ═══════════════════════════════════════════════════════════════════════════
# Conjecture test figure (5-panel per run, like reference)
# ═══════════════════════════════════════════════════════════════════════════

def fig_conjecture_test(all_specs, all_sgd, all_metrics, all_grok_steps, all_pca,
                        all_sgd_spikes=None):
    """Per-seed 5-panel conjecture test (PCA, rotation, matrix comm, SGD, acc)."""
    for seed in SEEDS:
        key = (1.0, seed)
        if key not in all_specs:
            continue
        spec = all_specs[key]
        steps = spec["steps"]
        sgd = all_sgd.get(key, np.zeros_like(steps))
        pca = all_pca.get(key, {})
        _, test_acc = get_metrics_at_steps(all_metrics[key], steps)
        gs = all_grok_steps.get(key)
        spike = (all_sgd_spikes or {}).get(key)

        has_sgd = sgd.max() > 0
        has_pca = "pca_steps" in pca and len(pca["pca_steps"]) > 0
        n_panels = 3 + int(has_sgd) + int(has_pca)
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.2 * n_panels), sharex=True)
        fig.subplots_adjust(hspace=0.15)
        pc = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
        panel = 0

        # P1: PCA eigenvalue ratios (if available)
        if has_pca:
            ax = axes[panel]
            for i in range(min(3, pca["explained"].shape[1])):
                ax.plot(pca["pca_steps"], pca["explained"][:, i], color=pc[i],
                        label=f"$\\lambda_{i+1}/\\Sigma$", lw=1.5)
            if gs: ax.axvline(gs, color="green", ls=":", alpha=0.5, lw=1)
            ax.set_ylabel("Explained var. ratio\n(expanding window)")
            ax.legend(fontsize=7, loc="upper right")
            ax.set_title(f"Dyck Conjecture Test: seed={seed}, layer {LAYER_IDX}", fontsize=12)
            panel += 1

        # P2: SVD gaps + commutator
        ax = axes[panel]
        ax.plot(steps, spec["g12_q"], color="#e41a1c", lw=2, label="$\\sigma_1-\\sigma_2$ ($W_Q$)")
        ax.plot(steps, spec["g23_q"], color="#ff7f0e", lw=1.5, label="$\\sigma_2-\\sigma_3$ ($W_Q$)")
        ax.plot(steps, spec["g12_k"], color="#e41a1c", lw=1, ls="--", alpha=0.5, label="$\\sigma_1-\\sigma_2$ ($W_K$)")
        if gs: ax.axvline(gs, color="green", ls=":", alpha=0.5, lw=1)
        ax.set_ylabel("SVD gaps")
        ax.legend(fontsize=7, loc="upper right")
        if not has_pca:
            ax.set_title(f"Dyck Conjecture Test: seed={seed}, layer {LAYER_IDX}", fontsize=12)
        panel += 1

        # P3: Matrix commutator
        ax = axes[panel]
        ax.plot(steps, spec["comm_norms"], color="#d62728", lw=2,
                label="$\\|[W_Q, W_K]\\|_F$")
        if gs: ax.axvline(gs, color="green", ls=":", alpha=0.5, lw=1)
        # Find and mark peak
        skip = min(3, len(spec["comm_norms"]) - 1)
        peak_idx = skip + np.argmax(spec["comm_norms"][skip:])
        ax.axvline(steps[peak_idx], color="orange", ls="--", alpha=0.7, lw=1,
                   label=f"peak @{steps[peak_idx]}")
        ax.set_ylabel("Matrix commutator")
        ax.legend(fontsize=7, loc="upper left")
        panel += 1

        # P4: SGD defect (if available)
        if has_sgd:
            ax = axes[panel]
            ax.semilogy(steps, np.maximum(sgd, 1e-10), color="#9467bd", lw=2,
                        label="SGD defect $D$")
            if gs: ax.axvline(gs, color="green", ls=":", alpha=0.5, lw=1)
            ax.set_ylabel("SGD defect (log)")
            ax.legend(fontsize=7, loc="upper left")
            panel += 1

        # P5: Generalization
        ax = axes[panel]
        train_acc, _ = get_metrics_at_steps(all_metrics[key], steps)
        m_steps = np.array([m["step"] for m in all_metrics[key]])
        m_train = np.array([m["train_acc"] for m in all_metrics[key]])
        m_test = np.array([m["test_acc"] for m in all_metrics[key]])
        ax.plot(m_steps, m_train, color="#1f77b4", lw=1.5, label="Train")
        ax.plot(m_steps, m_test, color="#ff7f0e", lw=1.5, label="Test")
        if gs: ax.axvline(gs, color="green", ls=":", alpha=0.5, lw=1,
                           label=f"grok @{gs}")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Training step")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=7, loc="center right")

        # Event timeline at bottom
        events = find_events(spec, sgd, all_metrics[key], gs, spike)
        event_order = [("g23_peak", "g₂₃↓"), ("g12_min", "σ₁≈σ₂"),
                       ("sgd_spike", "SGD↑"), ("comm_peak", "comm peak"),
                       ("g12_max", "σ₁≫σ₂"), ("comm_collapse", "comm↓"),
                       ("grok", "grok")]
        parts = [f"{label}@{events[k]}" for k, label in event_order if k in events]
        fig.text(0.5, 0.005, "Timeline: " + " → ".join(parts) if parts else "—",
                 ha="center", fontsize=9, style="italic",
                 bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

        save_path = OUT_DIR / f"figCONJ_dyck_s{seed}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved figCONJ_dyck_s{seed}")


def print_verdict(all_specs, all_sgd, all_metrics, all_grok_steps,
                  all_sgd_spikes=None):
    """Print conjecture verification verdict."""
    print("\n" + "=" * 72)
    print("CONJECTURE VERIFICATION — DYCK")
    print("=" * 72)

    n = 0
    n_comm_before = 0
    n_sgd_before = 0
    comm_leads, sgd_leads = [], []

    for seed in SEEDS:
        key = (1.0, seed)
        if key not in all_specs:
            continue
        gs = all_grok_steps.get(key)
        if gs is None:
            continue
        n += 1
        spec = all_specs[key]
        sgd = all_sgd.get(key, np.zeros_like(spec["steps"]))
        spike = (all_sgd_spikes or {}).get(key)
        events = find_events(spec, sgd, all_metrics[key], gs, spike)

        comm_peak = events.get("comm_peak")
        sgd_spike = events.get("sgd_spike")

        if comm_peak is not None and comm_peak <= gs:
            n_comm_before += 1
            comm_leads.append(gs - comm_peak)
        if sgd_spike is not None and sgd_spike <= gs:
            n_sgd_before += 1
            sgd_leads.append(gs - sgd_spike)

        # Print timeline
        sorted_ev = sorted(events.items(), key=lambda x: x[1])
        parts = [f"{k}@{v}" for k, v in sorted_ev]
        print(f"  seed={seed}: {' → '.join(parts)}")

    print(f"\n{'─' * 72}")
    if n > 0:
        print(f"Matrix comm peak ≤ grok: {n_comm_before}/{n} ({100*n_comm_before/n:.0f}%)")
        if comm_leads:
            cl = np.array(comm_leads)
            print(f"  Lead time: mean={cl.mean():.0f}, median={np.median(cl):.0f}")
        print(f"SGD defect spike ≤ grok: {n_sgd_before}/{n} ({100*n_sgd_before/n:.0f}%)")
        if sgd_leads:
            sl = np.array(sgd_leads)
            print(f"  Lead time: mean={sl.mean():.0f}, median={np.median(sl):.0f}")
    else:
        print("No grokking runs to verify.")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("Dyck Spectral Geometry Analysis")
    print("=" * 50)

    # Load SGD defect data
    print("Loading SGD commutator defect data...")
    sgd_data = load_sgd_defect()

    all_specs = {}
    all_sgd = {}
    all_sgd_spikes = {}
    all_metrics = {}
    all_grok_steps = {}
    all_head_data = {}

    for wd in WDS:
        for seed in SEEDS:
            key = (wd, seed)
            tag = f"wd={wd}, seed={seed}"
            path = SWEEP_DIR / f"dyck_wd{wd}_s{seed}.pt"
            if not path.exists():
                print(f"  {tag}: checkpoint not found, skipping")
                continue

            print(f"  Processing {tag}...")
            data = load_run(wd, seed)

            # Spectral quantities
            spec = compute_spectral_quantities(data["attn_logs"], LAYER_IDX)
            all_specs[key] = spec

            # Per-head
            hd = compute_per_head(data["attn_logs"], LAYER_IDX)
            all_head_data[key] = hd

            # Metrics
            all_metrics[key] = data["metrics"]

            # SGD defect (interpolated for plotting, native spike detection)
            sgd = get_sgd_defect_at_steps(sgd_data, wd, seed, spec["steps"])
            all_sgd[key] = sgd
            all_sgd_spikes[key] = find_sgd_spike_step(sgd_data, wd, seed)

            # Grok step — use higher threshold for Dyck since test_acc
            # crosses 0.9 almost immediately (not classic delayed grokking).
            # 0.95 better captures the generalization improvement phase.
            gs = find_grok_step(data["metrics"], threshold=0.95)
            all_grok_steps[key] = gs
            grok_str = f"step {gs}" if gs else "N/A"
            print(f"    {len(spec['steps'])} checkpoints, grok={grok_str}")

    # Generate all figures
    print("\nGenerating figures...")
    fig_SVD1(all_specs, all_sgd, all_metrics, all_grok_steps)
    fig_SVD2(all_specs, all_sgd)
    fig_SVD3(all_specs, all_metrics, all_grok_steps)
    fig_SVD4(all_head_data, all_grok_steps)
    fig_SVD5(all_specs, all_sgd, all_metrics, all_grok_steps)
    fig_SVD6(all_specs, all_metrics, all_grok_steps)
    fig_PP1(all_specs, all_sgd, all_metrics, all_grok_steps, all_sgd_spikes)
    fig_PP2(all_specs, all_sgd, all_metrics, all_grok_steps, all_sgd_spikes)
    fig_PP3(all_specs, all_sgd, all_metrics, all_grok_steps)
    fig_PP4(all_specs, all_grok_steps)

    # PCA rotation analysis
    all_pca = {}
    print("\nComputing PCA rotation analysis...")
    for wd in WDS:
        for seed in SEEDS:
            key = (wd, seed)
            path = SWEEP_DIR / f"dyck_wd{wd}_s{seed}.pt"
            if not path.exists():
                continue
            data = load_run(wd, seed)
            pca_steps, explained, rot_steps, rotations = compute_pca_and_rotation(
                data["attn_logs"], LAYER_IDX)
            all_pca[key] = {
                "pca_steps": pca_steps, "explained": explained,
                "rot_steps": rot_steps, "rotations": rotations,
            }
            print(f"  {key}: {len(pca_steps)} PCA windows, {len(rot_steps)} rotation points")

    # Conjecture test figure (per-run 5-panel like reference)
    fig_conjecture_test(all_specs, all_sgd, all_metrics, all_grok_steps, all_pca,
                        all_sgd_spikes)

    # Temporal ordering
    print_temporal_ordering(all_specs, all_sgd, all_metrics, all_grok_steps,
                            all_sgd_spikes)

    # Verdict
    print_verdict(all_specs, all_sgd, all_metrics, all_grok_steps, all_sgd_spikes)

    # Save computed data
    save_path = OUT_DIR / "dyck_spectral_geometry_results.pt"
    torch.save({
        "all_specs": all_specs,
        "all_sgd": all_sgd,
        "all_grok_steps": all_grok_steps,
        "all_head_data": all_head_data,
        "all_pca": all_pca,
    }, save_path)
    print(f"\nResults saved to {save_path}")
    print("Done!")


if __name__ == "__main__":
    main()
