#!/usr/bin/env python3
"""
Spectral geometry analysis for SCAN grokking.

Tests the conjecture: grokking is mediated by spectral symmetry-breaking
in attention weight matrices.

Adapted from dyck_spectral_geometry.py for the SCAN encoder-decoder model.
SCAN model: d_model=256, n_heads=4, n_layers=3 enc + 3 dec, d_head=64.

Analyzes both decoder self-attention (where generalization happens)
and cross-attention layers.

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
  CONJ  -- Per-seed conjecture test panels
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

SWEEP_DIR = Path(__file__).parent / "../scan_sweep_results"
COMM_DIR  = Path(__file__).parent / "figures"
OUT_DIR   = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = [42, 137, 2024]
WDS   = [1.0, 0.0]

# SCAN: 3 encoder layers (idx 0-2) + 3 decoder layers (idx 3-5)
# Decoder self-attention is typically where the mechanism lives
LAYER_IDX = 3        # first decoder layer (decoder_self)
N_HEADS = 4
D_MODEL = 256
D_HEAD  = D_MODEL // N_HEADS  # 64
SVD_TOPK = 5
SMOOTH_WINDOW = 3

# ═══════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_run(wd, seed):
    path = SWEEP_DIR / f"scan_wd{wd}_s{seed}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def load_sgd_defect():
    """Load SGD commutator defect from generalization dynamics results.

    Uses generalization_dynamics_results.pt (100-step resolution) rather than
    commutator_results.pt (200-step resolution) for accurate spike detection.
    """
    path = COMM_DIR / "scan_generalization_dynamics_results.pt"
    if not path.exists():
        path = COMM_DIR / "scan_commutator_results.pt"
        if not path.exists():
            return {}
        return {"_source": "commutator", "_data": torch.load(path, map_location="cpu", weights_only=False)}
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {"_source": "gen_dynamics", "_data": data.get("all_runs", {})}


# ═══════════════════════════════════════════════════════════════════════════
# Core computations
# ═══════════════════════════════════════════════════════════════════════════

def compute_spectral_quantities(attn_logs, layer_idx=LAYER_IDX):
    """SVD of WQ, WK at each checkpoint; spectral gaps and commutator."""
    steps = []
    sv_Q, sv_K = [], []
    gap_Q, gap_K = [], []
    comm_norms = []
    head_gap_Q, head_comm_norms = [], []

    for snap in attn_logs:
        step = snap["step"]
        layer = snap["layers"][layer_idx]
        WQ = layer["WQ"].float().numpy().astype(np.float64)
        WK = layer["WK"].float().numpy().astype(np.float64)

        # Full matrix SVD
        SQ = np.linalg.svd(WQ, compute_uv=False)
        SK = np.linalg.svd(WK, compute_uv=False)

        k = min(SVD_TOPK, len(SQ))
        svq = np.zeros(SVD_TOPK); svq[:k] = SQ[:k]
        svk = np.zeros(SVD_TOPK); svk[:k] = SK[:k]
        sv_Q.append(svq)
        sv_K.append(svk)

        gap_Q.append([SQ[0] - SQ[1], SQ[1] - SQ[2]])
        gap_K.append([SK[0] - SK[1], SK[1] - SK[2]])

        # Matrix commutator
        comm = WQ @ WK - WK @ WQ
        comm_norms.append(np.linalg.norm(comm, "fro"))

        # Per-head
        hgq, hcn = [], []
        for h in range(N_HEADS):
            s, e = h * D_HEAD, (h + 1) * D_HEAD
            q_block = WQ[s:e, s:e]
            k_block = WK[s:e, s:e]
            sq_h = np.linalg.svd(q_block, compute_uv=False)
            hgq.append([sq_h[0] - sq_h[1], sq_h[1] - sq_h[2]])
            hcn.append(np.linalg.norm(q_block @ k_block - k_block @ q_block, "fro"))
        head_gap_Q.append(hgq)
        head_comm_norms.append(hcn)

        steps.append(step)

    return dict(
        steps=np.array(steps),
        sv_Q=np.array(sv_Q), sv_K=np.array(sv_K),
        gap_Q=np.array(gap_Q), gap_K=np.array(gap_K),
        comm_norms=np.array(comm_norms),
        head_gap_Q=np.array(head_gap_Q),   # [T, n_heads, 2]
        head_comm_norms=np.array(head_comm_norms),  # [T, n_heads]
    )


def compute_cross_attn_spectral(attn_logs, layer_idx=LAYER_IDX):
    """SVD and commutator for cross-attention (XWQ, XWK) if available."""
    steps = []
    gap_XQ, comm_x_norms = [], []

    for snap in attn_logs:
        layer = snap["layers"][layer_idx]
        if "XWQ" not in layer:
            return None

        XWQ = layer["XWQ"].float().numpy().astype(np.float64)
        XWK = layer["XWK"].float().numpy().astype(np.float64)

        SXQ = np.linalg.svd(XWQ, compute_uv=False)
        gap_XQ.append([SXQ[0] - SXQ[1], SXQ[1] - SXQ[2]])

        comm_x = XWQ @ XWK - XWK @ XWQ
        comm_x_norms.append(np.linalg.norm(comm_x, "fro"))
        steps.append(snap["step"])

    return dict(
        steps=np.array(steps),
        gap_XQ=np.array(gap_XQ),
        comm_x_norms=np.array(comm_x_norms),
    )


def compute_pca_and_rotation(attn_logs, layer_idx=LAYER_IDX, top_k=5):
    """Expanding-window PCA on QK update deltas + PC rotation angles."""
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
            for ii in range(k_rot):
                dot = np.clip(np.abs(np.dot(Vt[ii], prev_Vt[ii])), 0, 1)
                thetas.append(np.arccos(dot))
            rot_steps.append(step)
            rot_list.append(thetas)

        prev_Vt = Vt[:min(3, Vt.shape[0])].copy()

    return (np.array(pca_steps), np.array(explained_list),
            np.array(rot_steps), np.array(rot_list) if rot_list else np.empty((0, 3)))


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


def find_grok_step(metrics, threshold=0.95, key="test_seq_acc"):
    """Use test_seq_acc >= 0.95 for SCAN since token accuracy crosses 0.9
    almost immediately (not classic delayed grokking)."""
    for m in metrics:
        val = m.get(key, 0)
        if val >= threshold:
            return m["step"]
    return None


def get_metrics_at_steps(metrics, target_steps):
    m_steps = np.array([m["step"] for m in metrics])
    m_train = np.array([m["train_acc"] for m in metrics])
    m_test = np.array([m.get("test_acc", m.get("test_seq_acc", 0)) for m in metrics])
    train_acc = np.interp(target_steps, m_steps, m_train)
    test_acc = np.interp(target_steps, m_steps, m_test)
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
    """Detect SGD defect spike at native resolution (not interpolated)."""
    d_steps, d_vals = _get_sgd_raw(sgd_data, wd, seed)
    if d_steps is None or len(d_vals) < 3:
        return None
    baseline = np.median(d_vals[:3])
    baseline = max(baseline, 0.1)
    for i in range(2, len(d_vals)):
        if d_vals[i] > threshold_factor * baseline and d_vals[i] > min_defect:
            return int(d_steps[i])
    return None


def find_events(spec, sgd_defect, metrics, grok_step, sgd_spike_step=None):
    steps = spec["steps"]
    g12 = spec["gap_Q"][:, 0]
    g23 = spec["gap_Q"][:, 1]
    comm = spec["comm_norms"]
    events = {}

    # g23 peak in early training
    early = min(15, len(g23))
    events["g23_peak"] = steps[np.argmax(g23[:early])]

    # g12 minimum
    events["g12_min"] = steps[np.argmin(g12)]

    # SGD spike: use native-resolution detection if provided
    if sgd_spike_step is not None:
        events["sgd_spike"] = sgd_spike_step
    elif len(sgd_defect) >= 3:
        baseline = max(sgd_defect[0], 0.1)
        for i in range(1, len(sgd_defect)):
            d = sgd_defect[i]
            if d > 10 * baseline and d > 20:
                events["sgd_spike"] = steps[min(i, len(steps) - 1)]
                break

    # Commutator peak
    skip = min(3, len(comm) - 1)
    events["comm_peak"] = steps[skip + np.argmax(comm[skip:])]

    # g12 maximum
    events["g12_max"] = steps[skip + np.argmax(g12[skip:])]

    # Commutator collapse
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

def fig_SVD1(all_data, all_sgd, all_metrics, all_grok_steps):
    """5-panel timeseries for each (wd, seed) condition."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    titles = [r"$\sigma_1-\sigma_2$ ($W_Q$)", r"$\sigma_2-\sigma_3$ ($W_Q$)",
              r"$\|[W_Q,W_K]\|_F$", "SGD Defect", "Test Accuracy"]

    colors_wd1 = ["#e74c3c", "#c0392b", "#a93226"]
    colors_wd0 = ["#3498db", "#2980b9", "#2471a3"]

    for wd in WDS:
        for si, seed in enumerate(SEEDS):
            key = (wd, seed)
            if key not in all_data:
                continue
            sv = all_data[key]
            steps = sv["steps"]
            c = colors_wd1[si] if wd == 1.0 else colors_wd0[si]
            ls = "-" if wd == 1.0 else "--"
            label = f"wd={wd}, s={seed}"

            axes[0].plot(steps, sv["gap_Q"][:, 0], ls, color=c, label=label, alpha=0.8)
            axes[1].plot(steps, sv["gap_Q"][:, 1], ls, color=c, alpha=0.8)
            axes[2].plot(steps, sv["comm_norms"], ls, color=c, alpha=0.8)

            sgd = all_sgd.get(key, np.zeros_like(steps))
            axes[3].plot(steps, sgd, ls, color=c, alpha=0.8)

            _, test_acc = get_metrics_at_steps(all_metrics[key], steps)
            axes[4].plot(steps, test_acc, ls, color=c, alpha=0.8)

            gs = all_grok_steps.get(key)
            if gs is not None:
                for ax in axes:
                    ax.axvline(gs, color=c, alpha=0.3, ls=":")

    for ax, title in zip(axes, titles):
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Training step")
    fig.suptitle(f"SCAN: Spectral Quantities (decoder layer {LAYER_IDX-3})", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figSVD1_scan_timeseries.png", dpi=150)
    plt.close(fig)
    print("  saved figSVD1")


def fig_SVD2(all_data, all_sgd):
    """Scatter: SVD gaps vs both commutators."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Collect
    g12_mc, g23_mc, g12K_mc = [], [], []
    g12_sgd, g23_sgd = [], []

    for wd in WDS:
        for seed in SEEDS:
            key = (wd, seed)
            if key not in all_data:
                continue
            sv = all_data[key]
            sgd = all_sgd.get(key, np.zeros_like(sv["steps"]))
            marker = "o" if wd == 1.0 else "x"
            color = "#e74c3c" if wd == 1.0 else "#3498db"

            for i in range(len(sv["steps"])):
                g12_mc.append((sv["gap_Q"][i, 0], sv["comm_norms"][i], wd, seed))
                g23_mc.append((sv["gap_Q"][i, 1], sv["comm_norms"][i], wd, seed))
                g12K_mc.append((sv["gap_K"][i, 0], sv["comm_norms"][i], wd, seed))
                if sgd[i] > 0:
                    g12_sgd.append((sv["gap_Q"][i, 0], sgd[i], wd, seed))
                    g23_sgd.append((sv["gap_Q"][i, 1], sgd[i], wd, seed))

    def do_scatter(ax, data, xlabel, ylabel, title, logy=False):
        if not data:
            ax.set_title(title + " (no data)")
            return
        for wd_val, color, label in [(1.0, "#e74c3c", "wd=1"), (0.0, "#3498db", "wd=0")]:
            pts = [(x, y) for x, y, w, s in data if w == wd_val]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=10, alpha=0.35, color=color, label=label)
                if len(xs) > 2:
                    r = np.corrcoef(xs, np.log10(np.maximum(ys, 1e-10)) if logy else ys)[0, 1]
                    ax.annotate(f"{label}: r={r:.2f}", xy=(0.02, 0.95 - 0.08 * [1.0, 0.0].index(wd_val)),
                                xycoords="axes fraction", fontsize=8, color=color)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)

    do_scatter(axes[0, 0], g12_mc, r"$\sigma_1-\sigma_2$ ($W_Q$)", r"$\|[W_Q,W_K]\|_F$",
               r"$W_Q$ gap $\sigma_1-\sigma_2$ vs comm")
    do_scatter(axes[0, 1], g23_mc, r"$\sigma_2-\sigma_3$ ($W_Q$)", r"$\|[W_Q,W_K]\|_F$",
               r"$W_Q$ gap $\sigma_2-\sigma_3$ vs comm")
    do_scatter(axes[0, 2], g12K_mc, r"$\sigma_1-\sigma_2$ ($W_K$)", r"$\|[W_Q,W_K]\|_F$",
               r"$W_K$ gap $\sigma_1-\sigma_2$ vs comm")
    do_scatter(axes[1, 0], g12_sgd, r"$\sigma_1-\sigma_2$ ($W_Q$)", "SGD defect $D$",
               r"$W_Q$ gap vs SGD defect", logy=True)
    do_scatter(axes[1, 1], g23_sgd, r"$\sigma_2-\sigma_3$ ($W_Q$)", "SGD defect $D$",
               r"$W_Q$ gap₂₃ vs SGD defect", logy=True)
    axes[1, 2].set_visible(False)

    fig.suptitle("SCAN: Weight SVD gaps vs commutators", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figSVD2_scan_scatter.png", dpi=150)
    plt.close(fig)
    print("  saved figSVD2")


def fig_SVD3(all_data, all_metrics, all_grok_steps):
    """Phase-colored scatter (pre/trans/post grok)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    phase_colors = {"pre": "#3498db", "trans": "#f39c12", "post": "#2ecc71"}

    pre_12, trans_12, post_12 = [], [], []
    pre_23, trans_23, post_23 = [], [], []

    for seed in SEEDS:
        key = (1.0, seed)
        if key not in all_data:
            continue
        sv = all_data[key]
        gs = all_grok_steps.get(key)
        if gs is None:
            continue

        for i in range(len(sv["steps"])):
            step = sv["steps"][i]
            g12 = sv["gap_Q"][i, 0]
            g23 = sv["gap_Q"][i, 1]
            mc = sv["comm_norms"][i]
            if step < gs - 1000:
                pre_12.append((g12, mc)); pre_23.append((g23, mc))
            elif step < gs + 1000:
                trans_12.append((g12, mc)); trans_23.append((g23, mc))
            else:
                post_12.append((g12, mc)); post_23.append((g23, mc))

    for ax, pre, trans, post, xlabel in [
        (axes[0], pre_12, trans_12, post_12, r"$\sigma_1-\sigma_2$"),
        (axes[1], pre_23, trans_23, post_23, r"$\sigma_2-\sigma_3$"),
    ]:
        for data, phase, marker, alpha in [
            (pre, "pre", "o", 0.3), (trans, "trans", "D", 0.6), (post, "post", "s", 0.3)]:
            if data:
                xs, ys = zip(*data)
                ax.scatter(xs, ys, s=15, alpha=alpha, color=phase_colors[phase],
                          marker=marker, label=phase)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\|[W_Q,W_K]\|_F$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("SCAN: Phase-Colored Scatter (wd=1.0)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figSVD3_scan_phase_scatter.png", dpi=150)
    plt.close(fig)
    print("  saved figSVD3")


def fig_SVD4(all_data):
    """Per-head SVD gap vs per-head commutator."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    head_colors = plt.cm.Set1(np.linspace(0, 1, N_HEADS))

    for h in range(N_HEADS):
        ax = axes[h // 2, h % 2]
        for wd, marker, alpha in [(1.0, "o", 0.4), (0.0, "x", 0.3)]:
            for seed in SEEDS:
                key = (wd, seed)
                if key not in all_data:
                    continue
                sv = all_data[key]
                g12 = sv["head_gap_Q"][:, h, 0]
                cn = sv["head_comm_norms"][:, h]
                color = "#e74c3c" if wd == 1.0 else "#3498db"
                ax.scatter(g12, cn, s=10, alpha=alpha, color=color, marker=marker,
                          label=f"wd={wd}" if seed == 42 else "")
                if len(g12) > 2:
                    r = np.corrcoef(g12, cn)[0, 1]
                    if seed == 42:
                        y_off = 0 if wd == 1.0 else 0.08
                        ax.annotate(f"wd={wd}: r={r:.2f}",
                                    xy=(0.02, 0.95 - y_off),
                                    xycoords="axes fraction", fontsize=8, color=color)

        ax.set_xlabel(f"Head {h}: $\\sigma_1-\\sigma_2$")
        ax.set_ylabel(f"$\\|[W_Q^h, W_K^h]\\|_F$")
        ax.set_title(f"Head {h}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("SCAN: Per-Head SVD Gap vs Commutator", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figSVD4_scan_per_head.png", dpi=150)
    plt.close(fig)
    print("  saved figSVD4")


def fig_SVD5(all_data, all_sgd, all_metrics, all_grok_steps):
    """Narrative test: normalized [0,1] overlay."""
    grok_seeds = [s for s in SEEDS if all_grok_steps.get((1.0, s)) is not None]
    if not grok_seeds:
        print("  skipping figSVD5 (no grokking)")
        return

    fig, axes = plt.subplots(len(grok_seeds), 1, figsize=(14, 4 * len(grok_seeds)),
                              squeeze=False)

    for idx, seed in enumerate(grok_seeds):
        key = (1.0, seed)
        sv = all_data[key]
        steps = sv["steps"]
        sgd = all_sgd.get(key, np.zeros_like(steps))
        _, test_acc = get_metrics_at_steps(all_metrics[key], steps)
        gs = all_grok_steps[key]
        ax = axes[idx, 0]

        ax.plot(steps, norm01(sv["gap_Q"][:, 0]), "-", color="#e41a1c", lw=2,
                label=r"$\sigma_1-\sigma_2$")
        ax.plot(steps, norm01(sv["gap_Q"][:, 1]), "-", color="#ff7f0e", lw=2,
                label=r"$\sigma_2-\sigma_3$")
        ax.plot(steps, norm01(sv["comm_norms"]), "--", color="#9467bd", lw=2,
                label=r"$\|[W_Q,W_K]\|_F$")
        if sgd.max() > 0:
            log_sgd = np.log10(np.maximum(sgd, 1e-10))
            ax.plot(steps, norm01(log_sgd), "-.", color="#2ca02c", lw=1.5,
                    label="SGD defect (log)")
        ax.plot(steps, test_acc, ":", color="#17becf", lw=2, label="Test acc")
        ax.axvline(gs, color="black", ls="--", alpha=0.5, lw=1.5,
                   label=f"grok @{gs}")
        ax.set_ylabel(f"seed={seed}\nnormalized [0,1]")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Training step")
    fig.suptitle(r"SCAN: Narrative Test — $g_{23}\downarrow \to$ SGD$\uparrow \to$ "
                 r"modes compete $\to$ comm peak $\to$ mode wins $\to$ grok",
                 fontsize=11, y=1.01, style="italic")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figSVD5_scan_narrative.png", dpi=150)
    plt.close(fig)
    print("  saved figSVD5")


def fig_SVD6(all_data, all_metrics, all_grok_steps):
    """Grok vs control SVD dynamics."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    titles = [r"$\sigma_1-\sigma_2$ ($W_Q$)", r"$\|[W_Q,W_K]\|_F$", "Test Accuracy"]

    for wd in WDS:
        for seed in SEEDS:
            key = (wd, seed)
            if key not in all_data:
                continue
            sv = all_data[key]
            steps = sv["steps"]
            _, test_acc = get_metrics_at_steps(all_metrics[key], steps)
            c = "#e74c3c" if wd == 1.0 else "#3498db"
            ls = "-" if wd == 1.0 else "--"
            label = f"wd={wd}, s={seed}"

            axes[0].plot(steps, sv["gap_Q"][:, 0], ls, color=c, alpha=0.7, label=label)
            axes[1].plot(steps, sv["comm_norms"], ls, color=c, alpha=0.7)
            axes[2].plot(steps, test_acc, ls, color=c, alpha=0.7)

    for ax, t in zip(axes, titles):
        ax.set_ylabel(t); ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Training step")
    fig.suptitle("SCAN: Grok (wd=1.0) vs Control (wd=0.0)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figSVD6_scan_grok_vs_control.png", dpi=150)
    plt.close(fig)
    print("  saved figSVD6")


def _phase_portrait_ax(ax, sv, sgd_defect, metrics, grok_step,
                        color_by="step", title="", show_events=True,
                        sgd_spike_step=None):
    steps = sv["steps"]
    g12 = sv["gap_Q"][:, 0]
    comm = sv["comm_norms"]

    g12_s = rolling_mean(g12, SMOOTH_WINDOW)
    comm_s = rolling_mean(comm, SMOOTH_WINDOW)

    ax.plot(g12, comm, color="gray", alpha=0.15, lw=0.5, zorder=1)

    if color_by == "step":
        cmap = plt.cm.viridis
        cnorm = Normalize(vmin=steps.min(), vmax=steps.max())
        colors = cmap(cnorm(steps))
        cbar_label = "Training step"
    else:
        _, test_acc = get_metrics_at_steps(metrics, steps)
        cmap = plt.cm.RdYlGn
        cnorm = Normalize(vmin=0, vmax=1)
        colors = cmap(cnorm(test_acc))
        cbar_label = "Test accuracy"

    for i in range(len(steps) - 1):
        ax.plot(g12_s[i:i+2], comm_s[i:i+2], color=colors[i], lw=2, zorder=2)

    arrow_every = max(1, len(steps) // 15)
    for i in range(0, len(steps) - 1, arrow_every):
        dx = g12_s[min(i+1, len(g12_s)-1)] - g12_s[i]
        dy = comm_s[min(i+1, len(comm_s)-1)] - comm_s[i]
        if abs(dx) > 1e-10 or abs(dy) > 1e-10:
            ax.annotate("", xy=(g12_s[i] + dx * 0.5, comm_s[i] + dy * 0.5),
                        xytext=(g12_s[i], comm_s[i]),
                        arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.5), zorder=3)

    if show_events:
        events = find_events(sv, sgd_defect, metrics, grok_step, sgd_spike_step)
        ax.scatter(g12_s[0], comm_s[0], c="gray", marker="o", s=80, zorder=5,
                  edgecolors="black", label="Init")
        if "sgd_spike" in events:
            idx = np.argmin(np.abs(steps - events["sgd_spike"]))
            ax.scatter(g12_s[idx], comm_s[idx], c="green", marker="^", s=80,
                      zorder=5, edgecolors="black", label="SGD spike")
        if "comm_peak" in events:
            idx = np.argmin(np.abs(steps - events["comm_peak"]))
            ax.scatter(g12_s[idx], comm_s[idx], c="red", marker="D", s=80,
                      zorder=5, edgecolors="black", label="Comm peak")
        if "g12_min" in events:
            idx = np.argmin(np.abs(steps - events["g12_min"]))
            ax.scatter(g12_s[idx], comm_s[idx], c="purple", marker="v", s=80,
                      zorder=5, edgecolors="black", label=r"$\sigma_1\approx\sigma_2$")
        if grok_step is not None:
            idx = np.argmin(np.abs(steps - grok_step))
            if idx < len(g12_s):
                ax.scatter(g12_s[idx], comm_s[idx], c="orange", marker="*", s=200,
                          zorder=6, edgecolors="black", label="GROK")

    # Phase regions
    gap_thresh = np.percentile(g12_s, 40)
    comm_thresh = np.percentile(comm_s, 60)
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.axvspan(xlim[0], gap_thresh, alpha=0.045, color="purple", zorder=0)
    ax.fill_between([gap_thresh, xlim[1]], comm_thresh, ylim[1], alpha=0.045, color="red", zorder=0)
    ax.fill_between([gap_thresh, xlim[1]], ylim[0], comm_thresh, alpha=0.045, color="green", zorder=0)
    ax.axvline(gap_thresh, ls="--", color="gray", alpha=0.3, lw=0.8)
    ax.axhline(comm_thresh, ls="--", color="gray", alpha=0.3, lw=0.8)

    ax.set_xlabel(r"$\sigma_1 - \sigma_2$ (spectral gap)")
    ax.set_ylabel(r"$\|[W_Q, W_K]\|_F$ (non-commutativity)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=cbar_label, fraction=0.03, pad=0.02)


def fig_PP1(all_data, all_sgd, all_metrics, all_grok_steps, all_sgd_spikes=None):
    key = (1.0, 42)
    if key not in all_data:
        print("  skipping figPP1")
        return
    spike = (all_sgd_spikes or {}).get(key)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    _phase_portrait_ax(axes[0], all_data[key], all_sgd.get(key, np.zeros(1)),
                       all_metrics[key], all_grok_steps.get(key),
                       color_by="step", title="Colored by step",
                       sgd_spike_step=spike)
    axes[0].legend(fontsize=7, loc="upper left")
    _phase_portrait_ax(axes[1], all_data[key], all_sgd.get(key, np.zeros(1)),
                       all_metrics[key], all_grok_steps.get(key),
                       color_by="test_acc", title="Colored by test acc", show_events=False)
    fig.suptitle(f"SCAN: Hero Phase Portrait (wd=1.0, seed=42, dec layer {LAYER_IDX-3})", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPP1_scan_hero.png", dpi=150)
    plt.close(fig)
    print("  saved figPP1")


def fig_PP2(all_data, all_sgd, all_metrics, all_grok_steps, all_sgd_spikes=None):
    fig, axes = plt.subplots(len(WDS), len(SEEDS), figsize=(18, 12))
    for i, wd in enumerate(WDS):
        for j, seed in enumerate(SEEDS):
            key = (wd, seed)
            ax = axes[i, j]
            if key not in all_data:
                ax.set_visible(False)
                continue
            sgd = all_sgd.get(key, np.zeros_like(all_data[key]["steps"]))
            spike = (all_sgd_spikes or {}).get(key)
            _phase_portrait_ax(ax, all_data[key], sgd, all_metrics[key],
                              all_grok_steps.get(key), color_by="step",
                              title=f"wd={wd}, seed={seed}",
                              sgd_spike_step=spike)
            if i == 0 and j == 0:
                ax.legend(fontsize=6, loc="upper left")
    fig.suptitle("SCAN: Phase Portraits Grid", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPP2_scan_grid.png", dpi=150)
    plt.close(fig)
    print("  saved figPP2")


def fig_PP3(all_data, all_sgd, all_metrics, all_grok_steps):
    fig, ax = plt.subplots(figsize=(10, 8))
    for si, seed in enumerate(SEEDS):
        for wd, colors, ls_style, suffix in [
            (1.0, ["#e74c3c", "#c0392b", "#a93226"], "-", "grok"),
            (0.0, ["#3498db", "#2980b9", "#2471a3"], "--", "memo"),
        ]:
            key = (wd, seed)
            if key not in all_data:
                continue
            sv = all_data[key]
            g12_s = rolling_mean(sv["gap_Q"][:, 0], SMOOTH_WINDOW)
            comm_s = rolling_mean(sv["comm_norms"], SMOOTH_WINDOW)
            ax.plot(g12_s, comm_s, ls_style, color=colors[si], lw=2, alpha=0.8,
                    label=f"wd={wd}, s={seed} ({suffix})")
            ax.scatter(g12_s[0], comm_s[0], c="gray", marker="o", s=60, zorder=5)
            gs = all_grok_steps.get(key)
            if gs is not None:
                idx = np.argmin(np.abs(sv["steps"] - gs))
                if idx < len(g12_s):
                    ax.scatter(g12_s[idx], comm_s[idx], c="orange", marker="*",
                              s=150, zorder=6, edgecolors="black")

    ax.set_xlabel(r"$\sigma_1 - \sigma_2$ (spectral gap)")
    ax.set_ylabel(r"$\|[W_Q, W_K]\|_F$ (non-commutativity)")
    ax.set_title("SCAN: Grokking vs Memorizing Phase Trajectories")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPP3_scan_grok_vs_memo.png", dpi=150)
    plt.close(fig)
    print("  saved figPP3")


def fig_PP4(all_data, all_grok_steps):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    for wd in WDS:
        for si, seed in enumerate(SEEDS):
            key = (wd, seed)
            if key not in all_data:
                continue
            sv = all_data[key]
            g12 = rolling_mean(sv["gap_Q"][:, 0], SMOOTH_WINDOW)
            g23 = rolling_mean(sv["gap_Q"][:, 1], SMOOTH_WINDOW)
            comm = rolling_mean(sv["comm_norms"], SMOOTH_WINDOW)
            c = "#e74c3c" if wd == 1.0 else "#3498db"
            ls = "-" if wd == 1.0 else "--"
            ax.plot(g12, g23, comm, ls, color=c, alpha=0.7, label=f"wd={wd}, s={seed}", lw=1.5)
            ax.scatter(g12[0], g23[0], comm[0], c="gray", marker="o", s=50)
            gs = all_grok_steps.get(key)
            if gs is not None:
                idx = np.argmin(np.abs(sv["steps"] - gs))
                if idx < len(g12):
                    ax.scatter(g12[idx], g23[idx], comm[idx], c="orange", marker="*", s=150)
    ax.set_xlabel(r"$\sigma_1 - \sigma_2$")
    ax.set_ylabel(r"$\sigma_2 - \sigma_3$")
    ax.set_zlabel(r"$\|[W_Q, W_K]\|_F$")
    ax.set_title("SCAN: 3D Phase Portrait")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPP4_scan_3d.png", dpi=150)
    plt.close(fig)
    print("  saved figPP4")


def fig_conjecture_test(all_data, all_sgd, all_metrics, all_grok_steps, all_pca,
                        all_sgd_spikes=None):
    """Per-seed conjecture test panels."""
    for seed in SEEDS:
        key = (1.0, seed)
        if key not in all_data:
            continue
        sv = all_data[key]
        steps = sv["steps"]
        sgd = all_sgd.get(key, np.zeros_like(steps))
        pca = all_pca.get(key, {})
        gs = all_grok_steps.get(key)
        spike = (all_sgd_spikes or {}).get(key)

        has_sgd = sgd.max() > 0
        has_pca = "pca_steps" in pca and len(pca["pca_steps"]) > 0
        n_panels = 3 + int(has_sgd) + int(has_pca)
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.2 * n_panels), sharex=True)
        pc = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
        panel = 0

        if has_pca:
            ax = axes[panel]
            for i in range(min(3, pca["explained"].shape[1])):
                ax.plot(pca["pca_steps"], pca["explained"][:, i], color=pc[i],
                        label=f"$\\lambda_{i+1}/\\Sigma$", lw=1.5)
            if gs: ax.axvline(gs, color="green", ls=":", alpha=0.5)
            ax.set_ylabel("Explained var. ratio")
            ax.legend(fontsize=7)
            ax.set_title(f"SCAN Conjecture Test: seed={seed}, dec layer {LAYER_IDX-3}", fontsize=12)
            panel += 1

        ax = axes[panel]
        ax.plot(steps, sv["gap_Q"][:, 0], color="#e41a1c", lw=2, label=r"$\sigma_1-\sigma_2$ ($W_Q$)")
        ax.plot(steps, sv["gap_Q"][:, 1], color="#ff7f0e", lw=1.5, label=r"$\sigma_2-\sigma_3$ ($W_Q$)")
        if gs: ax.axvline(gs, color="green", ls=":", alpha=0.5)
        ax.set_ylabel("SVD gaps"); ax.legend(fontsize=7)
        if not has_pca:
            ax.set_title(f"SCAN Conjecture Test: seed={seed}, dec layer {LAYER_IDX-3}", fontsize=12)
        panel += 1

        ax = axes[panel]
        ax.plot(steps, sv["comm_norms"], color="#d62728", lw=2, label=r"$\|[W_Q, W_K]\|_F$")
        skip = min(3, len(sv["comm_norms"]) - 1)
        peak_idx = skip + np.argmax(sv["comm_norms"][skip:])
        ax.axvline(steps[peak_idx], color="orange", ls="--", alpha=0.7, label=f"peak @{steps[peak_idx]}")
        if gs: ax.axvline(gs, color="green", ls=":", alpha=0.5)
        ax.set_ylabel("Matrix commutator"); ax.legend(fontsize=7)
        panel += 1

        if has_sgd:
            ax = axes[panel]
            ax.semilogy(steps, np.maximum(sgd, 1e-10), color="#9467bd", lw=2, label="SGD defect")
            if gs: ax.axvline(gs, color="green", ls=":", alpha=0.5)
            ax.set_ylabel("SGD defect (log)"); ax.legend(fontsize=7)
            panel += 1

        ax = axes[panel]
        m_steps = np.array([m["step"] for m in all_metrics[key]])
        m_train = np.array([m["train_acc"] for m in all_metrics[key]])
        m_test = np.array([m.get("test_acc", m.get("test_seq_acc", 0)) for m in all_metrics[key]])
        ax.plot(m_steps, m_train, color="#1f77b4", lw=1.5, label="Train")
        ax.plot(m_steps, m_test, color="#ff7f0e", lw=1.5, label="Test")
        if gs: ax.axvline(gs, color="green", ls=":", alpha=0.5, label=f"grok @{gs}")
        ax.set_ylabel("Accuracy"); ax.set_xlabel("Step")
        ax.set_ylim(-0.05, 1.1); ax.legend(fontsize=7)

        events = find_events(sv, sgd, all_metrics[key], gs, spike)
        ev_order = [("g23_peak", "g₂₃↓"), ("g12_min", "σ₁≈σ₂"), ("sgd_spike", "SGD↑"),
                    ("comm_peak", "comm↑"), ("g12_max", "σ₁≫σ₂"), ("comm_collapse", "comm↓"),
                    ("grok", "grok")]
        parts = [f"{label}@{events[k]}" for k, label in ev_order if k in events]
        fig.text(0.5, 0.005, "Timeline: " + " → ".join(parts) if parts else "—",
                 ha="center", fontsize=9, style="italic",
                 bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

        plt.savefig(OUT_DIR / f"figCONJ_scan_s{seed}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved figCONJ_scan_s{seed}")


# ═══════════════════════════════════════════════════════════════════════════
# Temporal ordering and verdict
# ═══════════════════════════════════════════════════════════════════════════

def print_temporal_ordering(all_data, all_sgd, all_metrics, all_grok_steps,
                            all_sgd_spikes=None):
    predicted = ["g23_peak", "g12_min", "sgd_spike", "comm_peak", "g12_max", "grok"]
    predicted_labels = ["g₂₃↓", "σ₁≈σ₂", "D_SGD↑", "‖[Q,K]‖peak", "σ₁≫σ₂", "grok"]

    print("\n" + "=" * 70)
    print("TEMPORAL ORDERING OF KEY EVENTS (SCAN)")
    print("Predicted: " + " → ".join(predicted_labels))
    print("=" * 70)

    for seed in SEEDS:
        key = (1.0, seed)
        if key not in all_data:
            continue
        sv = all_data[key]
        sgd = all_sgd.get(key, np.zeros_like(sv["steps"]))
        gs = all_grok_steps.get(key)
        spike = (all_sgd_spikes or {}).get(key)

        events = find_events(sv, sgd, all_metrics[key], gs, spike)
        print(f"\n  wd=1.0, seed={seed}:")
        sorted_events = sorted(events.items(), key=lambda x: x[1])
        for name, step in sorted_events:
            label = predicted_labels[predicted.index(name)] if name in predicted else name
            print(f"    {label:>12s}: step {step:>6d}")

        actual_order = [name for name, _ in sorted_events if name in predicted]
        matches = sum(1 for a, b in zip(actual_order, predicted) if a == b)
        print(f"    Order match: {matches}/{len(predicted)} positions correct")
        print(f"    Actual: {' → '.join(predicted_labels[predicted.index(n)] for n in actual_order if n in predicted)}")


def print_verdict(all_data, all_sgd, all_metrics, all_grok_steps,
                  all_sgd_spikes=None):
    print("\n" + "=" * 72)
    print("CONJECTURE VERIFICATION — SCAN")
    print("=" * 72)

    n = 0
    n_comm_before = 0
    n_sgd_before = 0
    comm_leads, sgd_leads = [], []

    for seed in SEEDS:
        key = (1.0, seed)
        if key not in all_data:
            continue
        gs = all_grok_steps.get(key)
        if gs is None:
            continue
        n += 1
        sv = all_data[key]
        sgd = all_sgd.get(key, np.zeros_like(sv["steps"]))
        spike = (all_sgd_spikes or {}).get(key)
        events = find_events(sv, sgd, all_metrics[key], gs, spike)

        comm_peak = events.get("comm_peak")
        sgd_spike = events.get("sgd_spike")

        if comm_peak is not None and comm_peak <= gs:
            n_comm_before += 1
            comm_leads.append(gs - comm_peak)
        if sgd_spike is not None and sgd_spike <= gs:
            n_sgd_before += 1
            sgd_leads.append(gs - sgd_spike)

        sorted_ev = sorted(events.items(), key=lambda x: x[1])
        parts = [f"{k}@{v}" for k, v in sorted_ev]
        print(f"  seed={seed}: {' → '.join(parts)}")

    print(f"\n{'─' * 72}")
    if n > 0:
        print(f"Matrix comm peak <= grok: {n_comm_before}/{n} ({100*n_comm_before/n:.0f}%)")
        if comm_leads:
            cl = np.array(comm_leads)
            print(f"  Lead time: mean={cl.mean():.0f}, median={np.median(cl):.0f}")
        print(f"SGD defect spike <= grok: {n_sgd_before}/{n} ({100*n_sgd_before/n:.0f}%)")
        if sgd_leads:
            sl = np.array(sgd_leads)
            print(f"  Lead time: mean={sl.mean():.0f}, median={np.median(sl):.0f}")
    else:
        print("No grokking runs to verify.")


# ═══════════════════════════════════════════════════════════════════════════
# Multi-layer comparison
# ═══════════════════════════════════════════════════════════════════════════

def fig_multi_layer(data, sgd, metrics, grok_step, seed):
    """Compare spectral dynamics across all layers for a single run."""
    n_layers = len(data["attn_logs"][0]["layers"])
    fig, axes = plt.subplots(3, n_layers, figsize=(4 * n_layers, 10), squeeze=False)

    for li in range(n_layers):
        sv = compute_spectral_quantities(data["attn_logs"], layer_idx=li)
        layer_info = data["attn_logs"][0]["layers"][li]
        ltype = layer_info.get("type", "encoder")
        title = f"L{layer_info.get('layer', li)} ({ltype})"

        axes[0, li].plot(sv["steps"], sv["gap_Q"][:, 0], color="#e41a1c", lw=1.5)
        axes[0, li].set_title(title, fontsize=9)
        if li == 0: axes[0, li].set_ylabel(r"$\sigma_1-\sigma_2$")

        axes[1, li].plot(sv["steps"], sv["comm_norms"], color="#d62728", lw=1.5)
        if li == 0: axes[1, li].set_ylabel(r"$\|[W_Q,W_K]\|_F$")

        _, test_acc = get_metrics_at_steps(metrics, sv["steps"])
        axes[2, li].plot(sv["steps"], test_acc, color="#ff7f0e", lw=1.5)
        if li == 0: axes[2, li].set_ylabel("Test acc")
        axes[2, li].set_xlabel("Step")

        if grok_step:
            for row in range(3):
                axes[row, li].axvline(grok_step, color="green", ls=":", alpha=0.5)

    fig.suptitle(f"SCAN: All Layers (seed={seed}, wd=1.0)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"figML_scan_all_layers_s{seed}.png", dpi=150)
    plt.close(fig)
    print(f"  saved figML_scan_all_layers_s{seed}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("SCAN Spectral Geometry Analysis")
    print("=" * 50)

    # Load SGD defect
    print("Loading SGD commutator defect data...")
    sgd_data = load_sgd_defect()

    all_data = {}
    all_sgd = {}
    all_sgd_spikes = {}
    all_metrics = {}
    all_grok_steps = {}

    for wd in WDS:
        for seed in SEEDS:
            key = (wd, seed)
            tag = f"wd={wd}, seed={seed}"
            data = load_run(wd, seed)
            if data is None:
                print(f"  {tag}: not found, skipping")
                continue

            print(f"  Processing {tag}...")
            sv = compute_spectral_quantities(data["attn_logs"], LAYER_IDX)
            all_data[key] = sv
            all_metrics[key] = data["metrics"]
            sgd = get_sgd_defect_at_steps(sgd_data, wd, seed, sv["steps"])
            all_sgd[key] = sgd
            all_sgd_spikes[key] = find_sgd_spike_step(sgd_data, wd, seed)

            gs = find_grok_step(data["metrics"])
            all_grok_steps[key] = gs
            print(f"    {len(sv['steps'])} checkpoints, grok={f'step {gs}' if gs else 'N/A'}")

    # Generate all figures
    print("\nGenerating figures...")
    fig_SVD1(all_data, all_sgd, all_metrics, all_grok_steps)
    fig_SVD2(all_data, all_sgd)
    fig_SVD3(all_data, all_metrics, all_grok_steps)
    fig_SVD4(all_data)
    fig_SVD5(all_data, all_sgd, all_metrics, all_grok_steps)
    fig_SVD6(all_data, all_metrics, all_grok_steps)
    fig_PP1(all_data, all_sgd, all_metrics, all_grok_steps, all_sgd_spikes)
    fig_PP2(all_data, all_sgd, all_metrics, all_grok_steps, all_sgd_spikes)
    fig_PP3(all_data, all_sgd, all_metrics, all_grok_steps)
    fig_PP4(all_data, all_grok_steps)

    # PCA rotation
    all_pca = {}
    print("\nComputing PCA rotation analysis...")
    for wd in WDS:
        for seed in SEEDS:
            key = (wd, seed)
            data = load_run(wd, seed)
            if data is None:
                continue
            pca_steps, explained, rot_steps, rotations = compute_pca_and_rotation(
                data["attn_logs"], LAYER_IDX)
            all_pca[key] = {
                "pca_steps": pca_steps, "explained": explained,
                "rot_steps": rot_steps, "rotations": rotations,
            }
            print(f"  {key}: {len(pca_steps)} PCA windows")

    fig_conjecture_test(all_data, all_sgd, all_metrics, all_grok_steps, all_pca,
                        all_sgd_spikes)

    # Multi-layer comparison for seed=42
    print("\nMulti-layer analysis...")
    for seed in [42]:
        data = load_run(1.0, seed)
        if data is not None:
            fig_multi_layer(data, all_sgd.get((1.0, seed), np.zeros(1)),
                           all_metrics.get((1.0, seed), []), all_grok_steps.get((1.0, seed)), seed)

    # Temporal ordering and verdict
    print_temporal_ordering(all_data, all_sgd, all_metrics, all_grok_steps,
                            all_sgd_spikes)
    print_verdict(all_data, all_sgd, all_metrics, all_grok_steps, all_sgd_spikes)

    # Save
    save_path = OUT_DIR / "scan_spectral_geometry_results.pt"
    torch.save({
        "all_data": all_data,
        "all_sgd": all_sgd,
        "all_grok_steps": all_grok_steps,
        "all_pca": all_pca,
    }, save_path)
    print(f"\nResults saved to {save_path}")
    print("Done!")


if __name__ == "__main__":
    main()
