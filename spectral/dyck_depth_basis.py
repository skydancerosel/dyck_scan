#!/usr/bin/env python3
"""
Depth-conditioned representation geometry for Dyck-1.

Analogous to fourier_proper_basis.py — the "proper basis" for Dyck
is stack depth (0..12). We analyze how the model encodes depth in
its representation space.

Key questions:
- Is depth linearly encoded (1D subspace)?
- How does depth geometry change at grokking?
- What's the PCA structure of depth centroids?
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

from dyck.grok_sweep import DyckTransformerLM, VOCAB_SIZE, build_depth_dataset, split_dataset
from spectral.fourier_functional_dyck import load_model_at_step, extract_hidden_reps

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"


def depth_conditioned_analysis(reps, depths, mask):
    """Analyze representation geometry conditioned on depth.

    Args:
        reps: [N, T, D] hidden representations
        depths: [N, T] depth labels (-100 for padding)
        mask: [N, T] boolean valid mask

    Returns dict with centroids, PCA, inter-depth distances, linear probe R².
    """
    N, T, D = reps.shape

    # Flatten to [N*T, D] and [N*T] keeping only valid positions
    reps_flat = reps.reshape(-1, D)
    depths_flat = depths.reshape(-1)
    mask_flat = mask.reshape(-1)

    valid = mask_flat.astype(bool)
    reps_valid = reps_flat[valid]  # [M, D]
    depths_valid = depths_flat[valid]  # [M]

    # Depth centroids
    unique_depths = np.unique(depths_valid)
    centroids = {}
    counts = {}
    for d in unique_depths:
        if d < 0:
            continue
        idx = depths_valid == d
        centroids[int(d)] = reps_valid[idx].mean(axis=0)
        counts[int(d)] = idx.sum()

    # PCA of centroids
    depth_keys = sorted(centroids.keys())
    centroid_matrix = np.stack([centroids[d] for d in depth_keys])  # [n_depths, D]
    pca = PCA(n_components=min(5, len(depth_keys)))
    centroid_pca = pca.fit_transform(centroid_matrix)
    explained = pca.explained_variance_ratio_

    # Inter-depth distances (consecutive depths)
    distances = []
    for i in range(len(depth_keys) - 1):
        d = np.linalg.norm(centroids[depth_keys[i+1]] - centroids[depth_keys[i]])
        distances.append(d)

    # Linear probe R² for depth
    probe = Ridge(alpha=1.0)
    probe.fit(reps_valid, depths_valid)
    pred = probe.predict(reps_valid)
    r2 = r2_score(depths_valid, pred)

    # PCA of all valid representations (colored by depth)
    pca_all = PCA(n_components=2)
    reps_pca = pca_all.fit_transform(reps_valid)

    return {
        "centroids": centroids,
        "counts": counts,
        "depth_keys": depth_keys,
        "centroid_pca": centroid_pca,
        "explained_var": explained,
        "inter_depth_distances": distances,
        "linear_r2": r2,
        "reps_pca": reps_pca,
        "depths_valid": depths_valid,
        "pca_all_explained": pca_all.explained_variance_ratio_,
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Generate test data
    X_all, Y_all = build_depth_dataset(n_seqs=5050, max_pairs=12, ctx_len=24, seed=0)
    _, _, test_x, test_y = split_dataset(X_all, Y_all, frac_train=50/5050, seed=0)
    test_x = test_x[:500]
    test_y = test_y[:500]
    mask = (test_y != -100).numpy()
    depths = test_y.numpy()

    # Analysis steps
    analysis_steps = [0, 200, 500, 1000, 2000, 5000]

    all_results = {}

    for tag, ckpt_name in [("grok", "dyck_grok_fourier.pt"), ("memo", "dyck_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        print(f"\n{'='*50}")
        print(f"Depth basis analysis: {tag}")
        print(f"{'='*50}")

        all_results[tag] = {}

        for target_step in analysis_steps:
            model, cfg, actual_step = load_model_at_step(ckpt_path, target_step)
            reps = extract_hidden_reps(model, test_x)

            print(f"\n  Step {actual_step}:")

            step_results = {}
            for layer_name in ["embedding", "layer_0", "layer_1"]:
                if layer_name not in reps:
                    continue
                result = depth_conditioned_analysis(reps[layer_name], depths, mask)
                step_results[layer_name] = result
                print(f"    {layer_name}: R²={result['linear_r2']:.3f}, "
                      f"PCA1_var={result['explained_var'][0]:.3f}, "
                      f"n_depths={len(result['depth_keys'])}")

            all_results[tag][actual_step] = step_results

    # ── Plot 1: Depth centroid PCA (grok vs memo, final step) ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Depth Centroids in PCA Space: Grokked vs Memorized", fontsize=14)

    for row, tag in enumerate(["grok", "memo"]):
        steps = sorted(all_results[tag].keys())
        late = steps[-1]
        for col, layer_name in enumerate(["embedding", "layer_0", "layer_1"]):
            ax = axes[row, col]
            r = all_results[tag][late][layer_name]

            # Plot centroid trajectory in PCA space
            pca_coords = r["centroid_pca"]
            depths_k = r["depth_keys"]
            sc = ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                           c=depths_k, cmap="viridis", s=80, zorder=5)
            # Connect consecutive depths with lines
            ax.plot(pca_coords[:, 0], pca_coords[:, 1], 'k-', alpha=0.3, linewidth=1)
            # Label depths
            for i, d in enumerate(depths_k):
                ax.annotate(str(d), (pca_coords[i, 0], pca_coords[i, 1]),
                           fontsize=8, ha='center', va='bottom')

            ax.set_title(f"{tag} step={late}: {layer_name}\n"
                         f"R²={r['linear_r2']:.3f}, PCA1={r['explained_var'][0]:.2f}")
            ax.set_xlabel(f"PC1 ({r['explained_var'][0]:.1%})")
            ax.set_ylabel(f"PC2 ({r['explained_var'][1]:.1%})")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "depth_centroids_pca.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {FIG_DIR / 'depth_centroids_pca.png'}")

    # ── Plot 2: All representations colored by depth ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Representations Colored by Depth: Grokked vs Memorized", fontsize=14)

    for row, tag in enumerate(["grok", "memo"]):
        steps = sorted(all_results[tag].keys())
        late = steps[-1]
        for col, layer_name in enumerate(["embedding", "layer_0", "layer_1"]):
            ax = axes[row, col]
            r = all_results[tag][late][layer_name]
            # Subsample for plotting
            n_plot = min(5000, len(r["reps_pca"]))
            idx = np.random.choice(len(r["reps_pca"]), n_plot, replace=False)
            sc = ax.scatter(r["reps_pca"][idx, 0], r["reps_pca"][idx, 1],
                           c=r["depths_valid"][idx], cmap="viridis",
                           s=2, alpha=0.3)
            plt.colorbar(sc, ax=ax, label="depth")
            ax.set_title(f"{tag} step={late}: {layer_name}\nR²={r['linear_r2']:.3f}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "depth_representations_scatter.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'depth_representations_scatter.png'}")

    # ── Plot 3: Linear R² over training ──
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_title("Depth Linear Probe R² During Training", fontsize=14)

    for tag, ls in [("grok", "-"), ("memo", "--")]:
        for layer_name, color in [("embedding", "gray"), ("layer_0", "steelblue"), ("layer_1", "darkred")]:
            steps_sorted = sorted(all_results[tag].keys())
            r2_vals = [all_results[tag][s][layer_name]["linear_r2"] for s in steps_sorted]
            ax.plot(steps_sorted, r2_vals, ls, color=color, label=f"{tag}/{layer_name}",
                    marker='o', markersize=3)

    ax.set_xlabel("Training step")
    ax.set_ylabel("R² (depth prediction)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "depth_probe_r2_training.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'depth_probe_r2_training.png'}")

    # ── Plot 4: Inter-depth distances ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Inter-Depth Representation Distances", fontsize=14)

    for i, tag in enumerate(["grok", "memo"]):
        ax = axes[i]
        steps = sorted(all_results[tag].keys())
        late = steps[-1]
        r = all_results[tag][late]["layer_1"]
        depths_k = r["depth_keys"]
        dists = r["inter_depth_distances"]
        ax.bar(range(len(dists)), dists, color="steelblue" if tag == "grok" else "coral")
        ax.set_xticks(range(len(dists)))
        ax.set_xticklabels([f"{depths_k[i]}→{depths_k[i+1]}" for i in range(len(dists))],
                          rotation=45, fontsize=7)
        ax.set_title(f"{tag} step={late} (layer_1)")
        ax.set_ylabel("||centroid(d+1) - centroid(d)||")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "inter_depth_distances.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'inter_depth_distances.png'}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Depth Basis Representation Geometry")
    print("="*60)
    for tag in ["grok", "memo"]:
        steps = sorted(all_results[tag].keys())
        late = steps[-1]
        print(f"\n{tag} (step {late}):")
        for layer_name in ["embedding", "layer_0", "layer_1"]:
            r = all_results[tag][late][layer_name]
            print(f"  {layer_name}: R²={r['linear_r2']:.3f}, "
                  f"PCA explained={r['explained_var'][:3].round(3)}, "
                  f"mean inter-depth dist={np.mean(r['inter_depth_distances']):.2f}")


if __name__ == "__main__":
    main()
