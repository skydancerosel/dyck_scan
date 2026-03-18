#!/usr/bin/env python3
"""
PCA eigenanalysis of Dyck-language grokking sweep results.

Task: Dyck-1 depth prediction (predict stack depth at each position).

Loads .pt files from dyck_sweep_results/ and produces:
  figA — PC1% bar chart: grok (wd=1.0) vs no-wd (wd=0.0)
  figB — PC1% heatmap: weight matrices × layers
  figC — Top-5 eigenspectrum per weight matrix
  figE — Z-scores above random-walk null model
  figF — Expanding-window PC1% over training
  Summary JSON
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
SWEEP_DIR = Path(__file__).parent / "../dyck_sweep_results"
OUT_DIR = Path(__file__).parent / "figures"
TOP_K = 10
ATTN_KEYS = ["WQ", "WK", "WV", "WO"]
MLP_KEYS  = ["W_up", "W_down"]
WEIGHT_KEYS = ATTN_KEYS + MLP_KEYS
COLORS_WK = {
    "WQ": "#1f77b4", "WK": "#ff7f0e", "WV": "#2ca02c", "WO": "#d62728",
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


def compute_all_pca(runs):
    all_pca = {}
    for (wd, seed), data in runs.items():
        n_layers = data["cfg"]["N_LAYERS"]
        logs = data["attn_logs"]
        if len(logs) < 5:
            print(f"  [skip] wd={wd} s={seed}: only {len(logs)} snapshots")
            continue
        for li in range(n_layers):
            for wkey in WEIGHT_KEYS:
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

def fig_a_grok_vs_nowd(all_pca, n_layers):
    """PC1% bar chart — grok vs no-wd."""
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5), squeeze=False)
    for col, li in enumerate(range(n_layers)):
        ax = axes[0, col]
        x = np.arange(len(WEIGHT_KEYS))
        width = 0.35

        grok_means, grok_stds = [], []
        nowd_means, nowd_stds = [], []

        for wkey in WEIGHT_KEYS:
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
        ax.set_title(f"Layer {li}")
        ax.set_xticks(x)
        ax.set_xticklabels(WEIGHT_KEYS)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Dyck Depth: PC1% — wd=1.0 vs wd=0.0\n(mean over 3 seeds)",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figA_dyck_grok_vs_nowd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figA_dyck_grok_vs_nowd.png")


def fig_b_heatmap(all_pca, n_layers):
    """PC1% heatmap — layers × weight matrices, wd=1.0."""
    fig, ax = plt.subplots(figsize=(6, 3 + n_layers))
    data_grid = np.zeros((n_layers, len(WEIGHT_KEYS)))

    for li in range(n_layers):
        for j, wkey in enumerate(WEIGHT_KEYS):
            vals = [all_pca[(1.0, s, li, wkey)]["explained_ratio"][0] * 100
                    for s in SEEDS if (1.0, s, li, wkey) in all_pca]
            data_grid[li, j] = np.mean(vals) if vals else 0

    im = ax.imshow(data_grid, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {i}" for i in range(n_layers)])
    ax.set_xticks(range(len(WEIGHT_KEYS)))
    ax.set_xticklabels(WEIGHT_KEYS)
    for i in range(n_layers):
        for j in range(len(WEIGHT_KEYS)):
            color = "white" if data_grid[i, j] > 60 else "black"
            ax.text(j, i, f"{data_grid[i,j]:.1f}", ha="center", va="center",
                    fontsize=10, color=color)
    fig.colorbar(im, ax=ax, pad=0.02).set_label("PC1 %")

    fig.suptitle("Dyck Depth: PC1% Heatmap (wd=1.0, mean over seeds)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figB_dyck_pc1_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figB_dyck_pc1_heatmap.png")


def fig_c_eigenspectrum(all_pca, n_layers):
    """Top-5 eigenspectrum per weight matrix (last layer)."""
    li = n_layers - 1
    fig, axes = plt.subplots(1, len(WEIGHT_KEYS), figsize=(4.5 * len(WEIGHT_KEYS), 4),
                             squeeze=False)
    for wi, wkey in enumerate(WEIGHT_KEYS):
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
        ax.set_title(wkey)
        ax.set_xticks(x)
        ax.set_xticklabels([f"PC{i+1}" for i in range(5)])
        if wi == 0:
            ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Top-5 Eigenspectrum (Layer {li}, Dyck Depth)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figC_dyck_eigenspectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figC_dyck_eigenspectrum.png")


def fig_e_null_model(runs, n_layers, n_null_trials=10):
    """Z-scores above random-walk null."""
    li = n_layers - 1
    print(f"\n  Computing random-walk null (layer {li})...")
    z_scores = {}

    for wkey in WEIGHT_KEYS:
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
                  f"null={null_mean:.1f}±{null_std:.1f}% → z={z:.1f}σ")
        z_scores[wkey] = np.mean(zs) if zs else 0

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(WEIGHT_KEYS))
    vals = [z_scores.get(wk, 0) for wk in WEIGHT_KEYS]
    ax.bar(x, vals, color=[COLORS_WK[wk] for wk in WEIGHT_KEYS], alpha=0.85)
    ax.axhline(y=3, color="gray", ls="--", alpha=0.5, label="3σ threshold")
    ax.set_ylabel("Z-score (σ above null)")
    ax.set_xticks(x)
    ax.set_xticklabels(WEIGHT_KEYS)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Dyck Depth: Sigma Above Random-Walk Null (Layer {li})", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figE_dyck_null_zscores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figE_dyck_null_zscores.png")


def fig_f_temporal(runs, n_layers):
    """Expanding-window PC1% over training."""
    li = n_layers - 1
    wkey = "WV"
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

    fig.suptitle(f"Dyck Depth: Expanding-Window PC1% (Layer {li}, {wkey})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figF_dyck_temporal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figF_dyck_temporal.png")


def fig_g_attn_vs_mlp(all_pca, n_layers):
    """Compare PC1% between attention and MLP weight matrices."""
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5), squeeze=False)

    for col, li in enumerate(range(n_layers)):
        ax = axes[0, col]
        categories = []
        means_grok, stds_grok = [], []
        means_nowd, stds_nowd = [], []

        for group_name, keys in [("Attn", ATTN_KEYS), ("MLP", MLP_KEYS)]:
            for wkey in keys:
                gvals = [all_pca[(1.0, s, li, wkey)]["explained_ratio"][0] * 100
                         for s in SEEDS if (1.0, s, li, wkey) in all_pca]
                nvals = [all_pca[(0.0, s, li, wkey)]["explained_ratio"][0] * 100
                         for s in SEEDS if (0.0, s, li, wkey) in all_pca]
                categories.append(wkey)
                means_grok.append(np.mean(gvals) if gvals else 0)
                stds_grok.append(np.std(gvals) if len(gvals) > 1 else 0)
                means_nowd.append(np.mean(nvals) if nvals else 0)
                stds_nowd.append(np.std(nvals) if len(nvals) > 1 else 0)

        if not categories:
            continue
        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, means_grok, width, yerr=stds_grok,
               label="wd=1.0", color="#2ca02c", alpha=0.85, capsize=3)
        ax.bar(x + width/2, means_nowd, width, yerr=stds_nowd,
               label="wd=0.0", color="#d62728", alpha=0.85, capsize=3)

        # Add vertical separator between attn and MLP
        n_attn = len(ATTN_KEYS)
        ax.axvline(x=n_attn - 0.5, color="gray", ls="--", alpha=0.4)
        ax.text(n_attn / 2 - 0.5, 95, "Attention", ha="center", fontsize=9, color="gray")
        ax.text(n_attn + len(MLP_KEYS) / 2 - 0.5, 95, "MLP", ha="center",
                fontsize=9, color="gray")

        ax.set_ylabel("PC1 explained var (%)")
        ax.set_title(f"Layer {li}")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha="right")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Dyck Depth: Attention vs MLP — PC1% Comparison\n(mean ± std over 3 seeds)",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figG_dyck_attn_vs_mlp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figG_dyck_attn_vs_mlp.png")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_summary_table(all_pca, runs, n_layers):
    summary = []
    print(f"\n{'='*80}")
    print("  DYCK DEPTH PCA SUMMARY")
    print(f"{'='*80}")

    for li in range(n_layers):
        print(f"\n  Layer {li}:")
        print(f"  {'wd':>4s}  {'seed':>5s}  {'grok':>5s}  ", end="")
        for wkey in WEIGHT_KEYS:
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
                for wkey in WEIGHT_KEYS:
                    k = (wd, seed, li, wkey)
                    if k in all_pca:
                        pc1 = all_pca[k]["explained_ratio"][0] * 100
                        pc1s.append(pc1)
                        print(f"{pc1:7.1f}%", end="  ")
                    else:
                        print(f"{'N/A':>8s}", end="  ")
                mean_pc1 = np.mean(pc1s) if pc1s else 0
                print(f"  {mean_pc1:5.1f}%")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)

    pt_files = list(SWEEP_DIR.glob("*.pt"))
    pt_files = [f for f in pt_files if "summary" not in f.stem]
    print(f"Found {len(pt_files)} .pt files in {SWEEP_DIR}/")

    if len(pt_files) == 0:
        print("No sweep results found. Run dyck_grok_sweep.py first.")
        sys.exit(1)

    print("Loading all runs...")
    runs = load_all_runs()
    print(f"Loaded {len(runs)} runs")

    first_run = next(iter(runs.values()))
    n_layers = first_run["cfg"]["N_LAYERS"]

    print("\nComputing PCA for all runs...")
    all_pca = compute_all_pca(runs)
    print(f"Computed {len(all_pca)} PCA results")

    print_summary_table(all_pca, runs, n_layers)

    print("\n" + "=" * 70)
    print("  GENERATING FIGURES")
    print("=" * 70)

    fig_a_grok_vs_nowd(all_pca, n_layers)
    fig_b_heatmap(all_pca, n_layers)
    fig_c_eigenspectrum(all_pca, n_layers)
    fig_e_null_model(runs, n_layers, n_null_trials=10)
    fig_f_temporal(runs, n_layers)

    # Check if MLP data is available
    has_mlp = any(k[3] in MLP_KEYS for k in all_pca)
    if has_mlp:
        fig_g_attn_vs_mlp(all_pca, n_layers)
    else:
        print("  [skip] figG: No MLP data in sweep results")

    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
