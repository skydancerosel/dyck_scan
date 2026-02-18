#!/usr/bin/env python3
"""
PCA eigenanalysis of SCAN grokking sweep results.

Task: SCAN command-to-action translation (seq2seq).

Loads .pt files from scan_sweep_results/ and produces:
  figA — PC1% bar chart: grok (wd=1.0) vs no-wd (wd=0.0)
  figB — PC1% heatmap: weight matrices × layers
  figC — Top-5 eigenspectrum per weight matrix
  figE — Z-scores above random-walk null model
  figF — Expanding-window PC1% over training
  figG — Attention vs MLP comparison
"""

import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── config ──────────────────────────────────────────────────────────────
SWEEP_DIR = Path(__file__).parent / "scan_sweep_results"
OUT_DIR = Path(__file__).parent / "scan_pca_plots"
TOP_K = 10

# Weight keys differ for encoder vs decoder in seq2seq
# Encoder layers: WQ, WK, WV, WO, W_up, W_down
# Decoder layers: WQ, WK, WV, WO, XWQ, XWK, XWV, XWO, W_up, W_down
ENCODER_ATTN_KEYS = ["WQ", "WK", "WV", "WO"]
ENCODER_MLP_KEYS = ["W_up", "W_down"]
DECODER_SELF_ATTN_KEYS = ["WQ", "WK", "WV", "WO"]
DECODER_CROSS_ATTN_KEYS = ["XWQ", "XWK", "XWV", "XWO"]
DECODER_MLP_KEYS = ["W_up", "W_down"]

# For analysis, we treat each layer entry as having these keys
ALL_WEIGHT_KEYS = ["WQ", "WK", "WV", "WO", "XWQ", "XWK", "XWV", "XWO",
                   "W_up", "W_down"]

COLORS_WK = {
    "WQ": "#1f77b4", "WK": "#ff7f0e", "WV": "#2ca02c", "WO": "#d62728",
    "XWQ": "#17becf", "XWK": "#bcbd22", "XWV": "#e377c2", "XWO": "#7f7f7f",
    "W_up": "#9467bd", "W_down": "#8c564b",
}
SEEDS = [42, 137, 2024]
WDS = [1.0, 0.0]


# ═══════════════════════════════════════════════════════════════════════════
# PCA helpers
# ═══════════════════════════════════════════════════════════════════════════

def collect_trajectory(attn_logs, layer_idx, key):
    """Return (steps, mats) for a single weight across all snapshots."""
    steps, mats = [], []
    for snap in attn_logs:
        if layer_idx >= len(snap["layers"]):
            return np.array([]), []
        layer_data = snap["layers"][layer_idx]
        if key not in layer_data:
            return np.array([]), []
        steps.append(snap["step"])
        mats.append(layer_data[key].float())
    return np.array(steps), mats


def pca_on_trajectory(mats, top_k):
    """PCA on flattened weight deltas from initialisation."""
    W0 = mats[0].reshape(-1).numpy()
    X = np.stack([m.reshape(-1).numpy() - W0 for m in mats])
    X -= X.mean(axis=0, keepdims=True)
    T, D = X.shape
    k = min(top_k, T, D)
    if T < 3:
        return None
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    eigenvalues = (S ** 2) / (T - 1)
    total_var = eigenvalues.sum()
    if total_var < 1e-15:
        return None
    explained = eigenvalues / total_var
    return {
        "eigenvalues": eigenvalues[:k],
        "explained_ratio": explained[:k],
        "components": Vt[:k],
        "total_var": float(total_var),
        "scores": (U[:, :k] * S[:k]),
    }


def expanding_window_pca(mats, top_k, n_checkpoints=15):
    """PCA on trajectory[:t] for growing t."""
    W0 = mats[0].reshape(-1).numpy()
    flat = np.stack([m.reshape(-1).numpy() - W0 for m in mats])
    T = len(flat)
    min_t = max(5, T // n_checkpoints)
    sizes = np.unique(np.linspace(min_t, T, n_checkpoints, dtype=int))
    records = []
    for t in sizes:
        chunk = flat[:t]
        chunk = chunk - chunk.mean(axis=0, keepdims=True)
        if chunk.shape[0] < 3:
            continue
        _, S, _ = np.linalg.svd(chunk, full_matrices=False)
        eig = (S ** 2) / (chunk.shape[0] - 1)
        total = eig.sum()
        if total < 1e-15:
            continue
        k = min(top_k, len(eig))
        records.append({
            "n_snaps": int(t),
            "pc1_pct": float(eig[0] / total) * 100,
            "top3_pct": float(eig[:min(3, k)].sum() / total) * 100,
        })
    return records


def random_walk_null(mats, n_trials=10, seed=123):
    """Random-walk null model — matched step norms, random directions."""
    flat = np.stack([m.reshape(-1).numpy() for m in mats])
    deltas = np.diff(flat, axis=0)
    step_norms = np.linalg.norm(deltas, axis=1)
    T, D = flat.shape
    rng = np.random.RandomState(seed)
    null_pc1 = []
    for _ in range(n_trials):
        directions = rng.randn(T - 1, D)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
        syn_deltas = directions * step_norms[:, None]
        syn_traj = np.zeros((T, D))
        syn_traj[1:] = np.cumsum(syn_deltas, axis=0)
        syn_traj -= syn_traj.mean(axis=0, keepdims=True)
        G = syn_traj @ syn_traj.T
        eigvals = np.linalg.eigvalsh(G)[::-1]
        eigvals = np.maximum(eigvals, 0)
        total = eigvals.sum()
        null_pc1.append(eigvals[0] / total if total > 0 else 0)
    return np.array(null_pc1)


# ═══════════════════════════════════════════════════════════════════════════
# Load
# ═══════════════════════════════════════════════════════════════════════════

def load_all_runs():
    runs = {}
    for pt_file in sorted(SWEEP_DIR.glob("*.pt")):
        if "summary" in pt_file.name:
            continue
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        cfg = data["cfg"]
        key = (cfg["WEIGHT_DECAY"], cfg["SEED"])
        runs[key] = data
    return runs


def get_layer_info(data):
    """Get information about layers from the first attn_log snapshot."""
    if not data["attn_logs"]:
        return []
    layers = data["attn_logs"][0]["layers"]
    info = []
    for li, layer_data in enumerate(layers):
        ltype = layer_data.get("type", "unknown")
        available_keys = [k for k in ALL_WEIGHT_KEYS if k in layer_data]
        info.append({"idx": li, "type": ltype, "keys": available_keys})
    return info


def compute_all_pca(runs):
    all_pca = {}
    for (wd, seed), data in runs.items():
        logs = data["attn_logs"]
        if len(logs) < 5:
            print(f"  [skip] wd={wd} s={seed}: only {len(logs)} snapshots")
            continue
        layer_info = get_layer_info(data)
        for li_info in layer_info:
            li = li_info["idx"]
            for wkey in li_info["keys"]:
                steps, mats = collect_trajectory(logs, li, wkey)
                if len(mats) == 0:
                    continue
                res = pca_on_trajectory(mats, TOP_K)
                if res is not None:
                    all_pca[(wd, seed, li, wkey)] = res
    return all_pca


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def fig_a_grok_vs_nowd(all_pca, layer_info):
    """PC1% bar chart — grok vs no-wd, per layer."""
    n_layers = len(layer_info)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5), squeeze=False)

    for col, li_info in enumerate(layer_info):
        ax = axes[0, col]
        li = li_info["idx"]
        wkeys = li_info["keys"]
        if not wkeys:
            continue

        x = np.arange(len(wkeys))
        width = 0.35
        grok_means, grok_stds = [], []
        nowd_means, nowd_stds = [], []

        for wkey in wkeys:
            gvals = [all_pca[(1.0, s, li, wkey)]["explained_ratio"][0] * 100
                     for s in SEEDS if (1.0, s, li, wkey) in all_pca]
            nvals = [all_pca[(0.0, s, li, wkey)]["explained_ratio"][0] * 100
                     for s in SEEDS if (0.0, s, li, wkey) in all_pca]
            grok_means.append(np.mean(gvals) if gvals else 0)
            grok_stds.append(np.std(gvals) if len(gvals) > 1 else 0)
            nowd_means.append(np.mean(nvals) if nvals else 0)
            nowd_stds.append(np.std(nvals) if len(nvals) > 1 else 0)

        ax.bar(x - width/2, grok_means, width, yerr=grok_stds,
               label="wd=1.0", color="#2ca02c", alpha=0.85, capsize=3)
        ax.bar(x + width/2, nowd_means, width, yerr=nowd_stds,
               label="wd=0.0", color="#d62728", alpha=0.85, capsize=3)
        ax.set_ylabel("Mean PC1 explained var (%)")
        ltype = li_info["type"]
        ax.set_title(f"L{li} ({ltype})")
        ax.set_xticks(x)
        ax.set_xticklabels(wkeys, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("SCAN: PC1% \u2014 wd=1.0 vs wd=0.0\n(mean over 3 seeds)",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figA_scan_grok_vs_nowd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figA_scan_grok_vs_nowd.png")


def fig_b_heatmap(all_pca, layer_info):
    """PC1% heatmap — layers × weight matrices, wd=1.0."""
    # Collect all unique weight keys across all layers
    all_keys = []
    for li_info in layer_info:
        for k in li_info["keys"]:
            if k not in all_keys:
                all_keys.append(k)

    n_layers = len(layer_info)
    fig, ax = plt.subplots(figsize=(max(8, len(all_keys)), 3 + n_layers))
    data_grid = np.zeros((n_layers, len(all_keys)))
    data_grid[:] = np.nan

    for li_info in layer_info:
        li = li_info["idx"]
        for j, wkey in enumerate(all_keys):
            if wkey not in li_info["keys"]:
                continue
            vals = [all_pca[(1.0, s, li, wkey)]["explained_ratio"][0] * 100
                    for s in SEEDS if (1.0, s, li, wkey) in all_pca]
            if vals:
                data_grid[li, j] = np.mean(vals)

    im = ax.imshow(data_grid, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{li['idx']} ({li['type'][:3]})" for li in layer_info],
                       fontsize=8)
    ax.set_xticks(range(len(all_keys)))
    ax.set_xticklabels(all_keys, fontsize=8, rotation=30, ha="right")
    for i in range(n_layers):
        for j in range(len(all_keys)):
            if not np.isnan(data_grid[i, j]):
                color = "white" if data_grid[i, j] > 60 else "black"
                ax.text(j, i, f"{data_grid[i,j]:.1f}", ha="center", va="center",
                        fontsize=8, color=color)
    fig.colorbar(im, ax=ax, pad=0.02).set_label("PC1 %")

    fig.suptitle("SCAN: PC1% Heatmap (wd=1.0, mean over seeds)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figB_scan_pc1_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figB_scan_pc1_heatmap.png")


def fig_c_eigenspectrum(all_pca, layer_info):
    """Top-5 eigenspectrum per weight matrix (last decoder layer)."""
    # Use last layer
    li_info = layer_info[-1]
    li = li_info["idx"]
    wkeys = li_info["keys"]

    fig, axes = plt.subplots(1, len(wkeys), figsize=(3.5 * len(wkeys), 4),
                             squeeze=False)
    for wi, wkey in enumerate(wkeys):
        ax = axes[0, wi]
        x = np.arange(5)
        for wd, color, label in [(1.0, "#2ca02c", "wd=1.0"), (0.0, "#d62728", "wd=0.0")]:
            vals_per_pc = []
            for pc_i in range(5):
                seed_vals = []
                for seed in SEEDS:
                    k = (wd, seed, li, wkey)
                    if k in all_pca and len(all_pca[k]["explained_ratio"]) > pc_i:
                        seed_vals.append(all_pca[k]["explained_ratio"][pc_i] * 100)
                vals_per_pc.append(np.mean(seed_vals) if seed_vals else 0)
            offset = -0.15 if wd == 1.0 else 0.15
            ax.bar(x + offset, vals_per_pc, 0.3, label=label, color=color, alpha=0.85)
        ax.set_xlabel("PC index")
        ax.set_ylabel("Explained var (%)")
        ax.set_title(wkey, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"PC{i+1}" for i in range(5)])
        if wi == 0:
            ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Top-5 Eigenspectrum (L{li}, SCAN)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figC_scan_eigenspectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figC_scan_eigenspectrum.png")


def fig_e_null_model(runs, layer_info, n_null_trials=10):
    """Z-scores above random-walk null."""
    # Use last layer
    li_info = layer_info[-1]
    li = li_info["idx"]
    wkeys = li_info["keys"]
    print(f"\n  Computing random-walk null (L{li})...")
    z_scores = {}

    for wkey in wkeys:
        zs = []
        for seed in SEEDS:
            run_key = (1.0, seed)
            if run_key not in runs:
                continue
            data = runs[run_key]
            logs = data["attn_logs"]
            if len(logs) < 5:
                continue
            _, mats = collect_trajectory(logs, li, wkey)
            if len(mats) == 0:
                continue
            res = pca_on_trajectory(mats, TOP_K)
            if res is None:
                continue
            real_pc1 = res["explained_ratio"][0] * 100
            null_dist = random_walk_null(mats, n_trials=n_null_trials, seed=seed) * 100
            null_mean = null_dist.mean()
            null_std = null_dist.std()
            z = (real_pc1 - null_mean) / (null_std + 1e-8)
            zs.append(z)
            print(f"    {wkey} s{seed}: real={real_pc1:.1f}% "
                  f"null={null_mean:.1f}+/-{null_std:.1f}% -> z={z:.1f}s")
        z_scores[wkey] = np.mean(zs) if zs else 0

    fig, ax = plt.subplots(figsize=(max(6, len(wkeys) * 0.8), 4))
    x = np.arange(len(wkeys))
    vals = [z_scores.get(wk, 0) for wk in wkeys]
    colors = [COLORS_WK.get(wk, "#333333") for wk in wkeys]
    ax.bar(x, vals, color=colors, alpha=0.85)
    ax.axhline(y=3, color="gray", ls="--", alpha=0.5, label="3s threshold")
    ax.set_ylabel("Z-score (s above null)")
    ax.set_xticks(x)
    ax.set_xticklabels(wkeys, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"SCAN: Sigma Above Random-Walk Null (L{li})", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figE_scan_null_zscores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figE_scan_null_zscores.png")


def fig_f_temporal(runs, layer_info):
    """Expanding-window PC1% over training."""
    li_info = layer_info[-1]
    li = li_info["idx"]
    wkey = "WV"  # Use self-attn WV as representative
    seed_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    for wi, wd in enumerate([1.0, 0.0]):
        ax = axes[0, wi]
        for si, seed in enumerate(SEEDS):
            run_key = (wd, seed)
            if run_key not in runs:
                continue
            data = runs[run_key]
            logs = data["attn_logs"]
            if len(logs) < 5:
                continue
            _, mats = collect_trajectory(logs, li, wkey)
            if len(mats) == 0:
                continue
            recs = expanding_window_pca(mats, TOP_K, n_checkpoints=20)
            if not recs:
                continue
            fracs = [r["n_snaps"] / len(mats) * 100 for r in recs]
            pc1s = [r["pc1_pct"] for r in recs]
            ax.plot(fracs, pc1s, color=seed_colors[si], linewidth=1.5,
                    label=f"s{seed}", marker=".", markersize=4)
        ax.set_xlabel("% of trajectory used")
        ax.set_ylabel("PC1 (%)")
        ax.set_title(f"wd={wd}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 100)

    fig.suptitle(f"SCAN: Expanding-Window PC1% (L{li}, {wkey})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figF_scan_temporal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figF_scan_temporal.png")


def fig_g_attn_vs_mlp(all_pca, layer_info):
    """Compare PC1% between attention and MLP weight matrices."""
    n_layers = len(layer_info)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5), squeeze=False)

    for col, li_info in enumerate(layer_info):
        ax = axes[0, col]
        li = li_info["idx"]
        wkeys = li_info["keys"]

        attn_keys = [k for k in wkeys if k in
                     ["WQ", "WK", "WV", "WO", "XWQ", "XWK", "XWV", "XWO"]]
        mlp_keys = [k for k in wkeys if k in ["W_up", "W_down"]]
        all_keys = attn_keys + mlp_keys

        if not all_keys:
            continue

        categories = all_keys
        means_grok, stds_grok = [], []
        means_nowd, stds_nowd = [], []

        for wkey in categories:
            gvals = [all_pca[(1.0, s, li, wkey)]["explained_ratio"][0] * 100
                     for s in SEEDS if (1.0, s, li, wkey) in all_pca]
            nvals = [all_pca[(0.0, s, li, wkey)]["explained_ratio"][0] * 100
                     for s in SEEDS if (0.0, s, li, wkey) in all_pca]
            means_grok.append(np.mean(gvals) if gvals else 0)
            stds_grok.append(np.std(gvals) if len(gvals) > 1 else 0)
            means_nowd.append(np.mean(nvals) if nvals else 0)
            stds_nowd.append(np.std(nvals) if len(nvals) > 1 else 0)

        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, means_grok, width, yerr=stds_grok,
               label="wd=1.0", color="#2ca02c", alpha=0.85, capsize=3)
        ax.bar(x + width/2, means_nowd, width, yerr=stds_nowd,
               label="wd=0.0", color="#d62728", alpha=0.85, capsize=3)

        n_attn = len(attn_keys)
        if n_attn > 0 and mlp_keys:
            ax.axvline(x=n_attn - 0.5, color="gray", ls="--", alpha=0.4)
            ax.text(n_attn / 2 - 0.5, 95, "Attention", ha="center",
                    fontsize=9, color="gray")
            ax.text(n_attn + len(mlp_keys) / 2 - 0.5, 95, "MLP",
                    ha="center", fontsize=9, color="gray")

        ltype = li_info["type"]
        ax.set_ylabel("PC1 explained var (%)")
        ax.set_title(f"L{li} ({ltype})")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("SCAN: Attention vs MLP — PC1% Comparison\n"
                 "(mean +/- std over 3 seeds)", fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figG_scan_attn_vs_mlp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figG_scan_attn_vs_mlp.png")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_summary_table(all_pca, runs, layer_info):
    print(f"\n{'='*80}")
    print("  SCAN PCA SUMMARY")
    print(f"{'='*80}")

    for li_info in layer_info:
        li = li_info["idx"]
        ltype = li_info["type"]
        wkeys = li_info["keys"]
        print(f"\n  L{li} ({ltype}):")
        print(f"  {'wd':>4s}  {'seed':>5s}  {'grok':>5s}  ", end="")
        for wkey in wkeys:
            print(f"{wkey:>8s}", end="  ")
        print("  mean")

        for wd in WDS:
            for seed in SEEDS:
                run_key = (wd, seed)
                if run_key not in runs:
                    continue
                run = runs[run_key]
                grokked = "YES" if run["grokked"] else "no"
                pc1s = []
                print(f"  {wd:4.1f}  {seed:5d}  {grokked:>5s}  ", end="")
                for wkey in wkeys:
                    k = (wd, seed, li, wkey)
                    if k in all_pca:
                        pc1 = all_pca[k]["explained_ratio"][0] * 100
                        pc1s.append(pc1)
                        print(f"{pc1:7.1f}%", end="  ")
                    else:
                        print(f"{'N/A':>8s}", end="  ")
                mean_pc1 = np.mean(pc1s) if pc1s else 0
                print(f"  {mean_pc1:5.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)

    pt_files = list(SWEEP_DIR.glob("*.pt"))
    pt_files = [f for f in pt_files if "summary" not in f.stem]
    print(f"Found {len(pt_files)} .pt files in {SWEEP_DIR}/")

    if len(pt_files) == 0:
        print("No sweep results found. Run scan_grok_sweep.py first.")
        sys.exit(1)

    print("Loading all runs...")
    runs = load_all_runs()
    print(f"Loaded {len(runs)} runs")

    # Get layer info from first available run
    first_run = next(iter(runs.values()))
    layer_info = get_layer_info(first_run)
    print(f"Found {len(layer_info)} layers:")
    for li in layer_info:
        print(f"  L{li['idx']} ({li['type']}): {li['keys']}")

    print("\nComputing PCA for all runs...")
    all_pca = compute_all_pca(runs)
    print(f"Computed {len(all_pca)} PCA results")

    print_summary_table(all_pca, runs, layer_info)

    print("\n" + "=" * 70)
    print("  GENERATING FIGURES")
    print("=" * 70)

    fig_a_grok_vs_nowd(all_pca, layer_info)
    fig_b_heatmap(all_pca, layer_info)
    fig_c_eigenspectrum(all_pca, layer_info)
    fig_e_null_model(runs, layer_info, n_null_trials=10)
    fig_f_temporal(runs, layer_info)
    fig_g_attn_vs_mlp(all_pca, layer_info)

    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
