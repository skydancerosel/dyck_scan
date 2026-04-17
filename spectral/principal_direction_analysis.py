#!/usr/bin/env python3
"""
Principal direction (v1, v2, v3) interpretability analysis.

Questions:
1. What do the top SVD directions of W_Q, W_K actually represent?
2. Do they align with Fourier modes, depth encoding, or token structure?
3. How do edge (v1) vs bulk (v2, v3) directions change pre/at/post grok?
4. Can we project representations onto v1/v2/v3 and see what information each carries?

For each checkpoint (pre-grok, at-grok, post-grok):
  - Extract v1, v2, v3 from SVD of W_Q and W_K
  - Project hidden representations onto each direction
  - Probe each projection for depth (Dyck) / action semantics (SCAN)
  - Compute Fourier content of each projected subspace
  - Track alignment between directions across training steps
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from dyck.grok_sweep import DyckTransformerLM, VOCAB_SIZE, build_depth_dataset, split_dataset
from spectral.fourier_functional_dyck import load_model_at_step, extract_hidden_reps

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"


def extract_svd_directions(model, layer_idx=0, topk=5):
    """Extract top-k SVD directions of W_Q and W_K for a given layer.

    Returns:
        For each of W_Q, W_K:
        - singular_values: [topk]
        - U_directions: [topk, d_model] — left singular vectors (output space)
        - V_directions: [topk, d_model] — right singular vectors (input space)
    """
    layer = model.encoder.layers[layer_idx]
    attn = layer.self_attn
    d = attn.embed_dim

    if attn._qkv_same_embed_dim:
        Wq = attn.in_proj_weight[:d].detach().cpu().numpy()
        Wk = attn.in_proj_weight[d:2*d].detach().cpu().numpy()
    else:
        Wq = attn.q_proj_weight.detach().cpu().numpy()
        Wk = attn.k_proj_weight.detach().cpu().numpy()

    results = {}
    for name, W in [("WQ", Wq), ("WK", Wk)]:
        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        results[name] = {
            "singular_values": S[:topk],
            "U": U[:, :topk],   # [d_model, topk] — left singular vectors
            "Vh": Vh[:topk, :], # [topk, d_model] — right singular vectors
            "full_S": S,
        }

    return results


def project_reps_onto_directions(reps, directions):
    """Project representations onto SVD directions.

    Args:
        reps: [N, T, D] hidden representations
        directions: [k, D] direction vectors (rows)

    Returns:
        projections: [N, T, k] — scalar projections onto each direction
    """
    # reps: [N, T, D], directions: [k, D]
    # projections = reps @ directions.T → [N, T, k]
    return np.einsum('ntd,kd->ntk', reps, directions)


def probe_projection(projections, targets, mask, k_idx):
    """Probe a single projected dimension for target prediction.

    Args:
        projections: [N, T, K] projections
        targets: [N, T] target labels
        mask: [N, T] boolean mask
        k_idx: which projection dimension to use

    Returns R² of linear probe using just this single dimension.
    """
    N, T, K = projections.shape
    proj_flat = projections[:, :, k_idx].reshape(-1)
    tgt_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1).astype(bool)

    X = proj_flat[mask_flat].reshape(-1, 1)
    y = tgt_flat[mask_flat]

    n = len(X)
    perm = np.random.permutation(n)
    split = int(0.7 * n)

    probe = Ridge(alpha=1.0)
    probe.fit(X[perm[:split]], y[perm[:split]])
    pred = probe.predict(X[perm[split:]])
    return r2_score(y[perm[split:]], pred)


def probe_multi_projection(projections, targets, mask, k_indices):
    """Probe using multiple projected dimensions jointly."""
    N, T, K = projections.shape
    proj_flat = projections[:, :, k_indices].reshape(-1, len(k_indices))
    tgt_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1).astype(bool)

    X = proj_flat[mask_flat]
    y = tgt_flat[mask_flat]

    n = len(X)
    perm = np.random.permutation(n)
    split = int(0.7 * n)

    probe = Ridge(alpha=1.0)
    probe.fit(X[perm[:split]], y[perm[:split]])
    pred = probe.predict(X[perm[split:]])
    return r2_score(y[perm[split:]], pred)


def fourier_of_projection(projections, mask):
    """Compute DFT power spectrum of projected representations.

    Args:
        projections: [N, T, K]
        mask: [N, T]

    Returns: [K, T//2+1] power spectra per projection direction
    """
    N, T, K = projections.shape
    masked = projections * mask[:, :, None]
    F = np.fft.rfft(masked, axis=1)  # [N, T//2+1, K]
    power = np.mean(np.abs(F) ** 2, axis=0)  # [T//2+1, K]
    return power.T  # [K, T//2+1]


def direction_alignment(dirs_a, dirs_b):
    """Compute alignment matrix between two sets of directions.

    Returns |cos(angle)| between each pair: [ka, kb]
    """
    # dirs_a: [ka, D], dirs_b: [kb, D]
    # Normalize
    a_norm = dirs_a / np.linalg.norm(dirs_a, axis=1, keepdims=True)
    b_norm = dirs_b / np.linalg.norm(dirs_b, axis=1, keepdims=True)
    return np.abs(a_norm @ b_norm.T)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # Load test data
    X_all, Y_all = build_depth_dataset(n_seqs=5050, max_pairs=12, ctx_len=24, seed=0)
    _, _, test_x, test_y = split_dataset(X_all, Y_all, frac_train=50/5050, seed=0)
    test_x = test_x[:500]
    test_y = test_y[:500]
    mask = (test_y != -100).numpy()
    depths = test_y.numpy()
    tokens = test_x.numpy()

    # Phases: pre-grok, at-grok, post-grok, late
    phase_steps = {
        "pre_grok": 200,
        "at_grok": 600,
        "post_grok": 2000,
        "late": 5000,
    }

    all_results = {}

    for tag, ckpt_name in [("grok", "dyck_grok_fourier.pt"), ("memo", "dyck_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        print(f"\n{'='*60}")
        print(f"Principal direction analysis: {tag}")
        print(f"{'='*60}")

        all_results[tag] = {}

        for phase_name, target_step in phase_steps.items():
            model, cfg, actual_step = load_model_at_step(ckpt_path, target_step)
            reps = extract_hidden_reps(model, test_x)

            print(f"\n  {phase_name} (step {actual_step}):")

            phase_results = {}

            for layer_idx in range(2):
                layer_name = f"layer_{layer_idx}"
                svd = extract_svd_directions(model, layer_idx, topk=5)

                # Print singular values
                for wname in ["WQ", "WK"]:
                    S = svd[wname]["singular_values"]
                    gaps = [S[i] - S[i+1] for i in range(len(S)-1)]
                    print(f"    {layer_name} {wname}: σ = {S.round(3)}, gaps = {np.array(gaps).round(3)}")

                # Project layer output representations onto W_Q right singular vectors
                # These are the "input space" directions — what the attention queries look for
                if layer_name in reps:
                    layer_reps = reps[layer_name]

                    for wname in ["WQ", "WK"]:
                        Vh = svd[wname]["Vh"]  # [topk, d_model]
                        projections = project_reps_onto_directions(layer_reps, Vh)

                        # Probe each direction for depth
                        r2_per_dir = []
                        for k in range(min(5, projections.shape[2])):
                            r2 = probe_projection(projections, depths, mask, k)
                            r2_per_dir.append(r2)

                        # Probe v1 alone, v2+v3 (bulk), all 5
                        r2_v1 = r2_per_dir[0]
                        r2_bulk = probe_multi_projection(projections, depths, mask, [1, 2])
                        r2_all5 = probe_multi_projection(projections, depths, mask, list(range(5)))

                        print(f"    {layer_name} {wname} depth R²: "
                              f"v1={r2_v1:.3f}, v2+v3={r2_bulk:.3f}, all5={r2_all5:.3f}")

                        # Fourier content per direction
                        power = fourier_of_projection(projections, mask)
                        dominant_omegas = [np.argmax(power[k]) for k in range(min(5, power.shape[0]))]
                        print(f"    {layer_name} {wname} dominant ω per direction: {dominant_omegas}")

                        # Token discrimination: does direction separate open vs close?
                        open_mask = (tokens == 0) & mask.astype(bool)
                        close_mask = (tokens == 1) & mask.astype(bool)
                        token_sep = []
                        for k in range(min(3, projections.shape[2])):
                            proj_k = projections[:, :, k]
                            if open_mask.sum() > 0 and close_mask.sum() > 0:
                                open_mean = proj_k[open_mask].mean()
                                close_mean = proj_k[close_mask].mean()
                                open_std = proj_k[open_mask].std() + 1e-8
                                close_std = proj_k[close_mask].std() + 1e-8
                                # Cohen's d
                                pooled_std = np.sqrt((open_std**2 + close_std**2) / 2)
                                d = abs(open_mean - close_mean) / pooled_std
                                token_sep.append(d)
                            else:
                                token_sep.append(0)
                        print(f"    {layer_name} {wname} token separation (Cohen's d): "
                              f"v1={token_sep[0]:.2f}, v2={token_sep[1]:.2f}, v3={token_sep[2]:.2f}")

                        phase_results[f"{layer_name}_{wname}"] = {
                            "singular_values": svd[wname]["singular_values"].tolist(),
                            "depth_r2_per_dir": r2_per_dir,
                            "depth_r2_v1": r2_v1,
                            "depth_r2_bulk": r2_bulk,
                            "depth_r2_all5": r2_all5,
                            "dominant_omegas": dominant_omegas,
                            "token_separation": token_sep,
                        }

            # Direction alignment across W_Q and W_K
            if "layer_0" in reps:
                svd0 = extract_svd_directions(model, 0, topk=3)
                svd1 = extract_svd_directions(model, 1, topk=3)

                # Q-K alignment within each layer
                for li in range(2):
                    svd_l = extract_svd_directions(model, li, topk=3)
                    align = direction_alignment(svd_l["WQ"]["Vh"], svd_l["WK"]["Vh"])
                    print(f"    layer_{li} Q-K alignment (|cos|):")
                    print(f"      {align.round(3)}")
                    phase_results[f"layer_{li}_qk_alignment"] = align.tolist()

                # Cross-layer alignment (layer 0 vs layer 1)
                for wname in ["WQ", "WK"]:
                    align = direction_alignment(svd0[wname]["Vh"], svd1[wname]["Vh"])
                    print(f"    {wname} cross-layer alignment (|cos|):")
                    print(f"      {align.round(3)}")
                    phase_results[f"crosslayer_{wname}_alignment"] = align.tolist()

            all_results[tag][phase_name] = phase_results

    # ══════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════

    # ── Plot 1: Singular values across phases ──
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("Singular Values of W_Q, W_K Across Grokking Phases", fontsize=14)

    for row, tag in enumerate(["grok", "memo"]):
        col = 0
        for layer_idx in range(2):
            for wname in ["WQ", "WK"]:
                ax = axes[row, col]
                phases = list(phase_steps.keys())
                key = f"layer_{layer_idx}_{wname}"
                for k in range(5):
                    vals = []
                    for phase in phases:
                        r = all_results[tag][phase].get(key, {})
                        sv = r.get("singular_values", [0]*5)
                        vals.append(sv[k] if k < len(sv) else 0)
                    ax.plot(range(len(phases)), vals, 'o-', label=f"σ_{k+1}", markersize=5)
                ax.set_xticks(range(len(phases)))
                ax.set_xticklabels(phases, fontsize=7, rotation=30)
                ax.set_title(f"{tag} L{layer_idx} {wname}", fontsize=9)
                ax.set_ylabel("σ")
                if col == 0:
                    ax.legend(fontsize=6)
                col += 1

    plt.tight_layout()
    fig.savefig(FIG_DIR / "svd_directions_across_phases.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {FIG_DIR / 'svd_directions_across_phases.png'}")

    # ── Plot 2: Depth R² per direction (edge vs bulk) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Depth Information in Edge (v1) vs Bulk (v2+v3) Directions", fontsize=14)

    for row, tag in enumerate(["grok", "memo"]):
        for col, (layer_idx, wname) in enumerate([(1, "WQ"), (1, "WK")]):
            ax = axes[row, col]
            phases = list(phase_steps.keys())
            key = f"layer_{layer_idx}_{wname}"

            v1_vals = [all_results[tag][p].get(key, {}).get("depth_r2_v1", 0) for p in phases]
            bulk_vals = [all_results[tag][p].get(key, {}).get("depth_r2_bulk", 0) for p in phases]
            all5_vals = [all_results[tag][p].get(key, {}).get("depth_r2_all5", 0) for p in phases]

            x = np.arange(len(phases))
            w = 0.25
            ax.bar(x - w, v1_vals, w, color='steelblue', label='v1 (edge)')
            ax.bar(x, bulk_vals, w, color='coral', label='v2+v3 (bulk)')
            ax.bar(x + w, all5_vals, w, color='gray', label='all 5')

            ax.set_xticks(x)
            ax.set_xticklabels(phases, fontsize=8)
            ax.set_ylabel("R² (depth)")
            ax.set_title(f"{tag} L{layer_idx} {wname}")
            ax.legend(fontsize=7)
            ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "edge_vs_bulk_depth_r2.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'edge_vs_bulk_depth_r2.png'}")

    # ── Plot 3: Token separation per direction ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Token Separation (Open vs Close) in v1/v2/v3 Directions", fontsize=14)

    for row, tag in enumerate(["grok", "memo"]):
        for col, (layer_idx, wname) in enumerate([(0, "WQ"), (1, "WQ")]):
            ax = axes[row, col]
            phases = list(phase_steps.keys())
            key = f"layer_{layer_idx}_{wname}"

            for k, (label, color) in enumerate(
                [("v1", "steelblue"), ("v2", "coral"), ("v3", "gray")]):
                vals = [all_results[tag][p].get(key, {}).get("token_separation", [0,0,0])[k] for p in phases]
                ax.plot(range(len(phases)), vals, 'o-', color=color, label=label, markersize=6)

            ax.set_xticks(range(len(phases)))
            ax.set_xticklabels(phases, fontsize=8)
            ax.set_ylabel("Cohen's d (open vs close)")
            ax.set_title(f"{tag} L{layer_idx} {wname}")
            ax.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "token_separation_per_direction.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'token_separation_per_direction.png'}")

    # ── Plot 4: Q-K alignment across phases ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("W_Q - W_K Direction Alignment Across Phases", fontsize=14)

    for row, tag in enumerate(["grok", "memo"]):
        for col, layer_idx in enumerate([0, 1]):
            ax = axes[row, col]
            phases = list(phase_steps.keys())
            key = f"layer_{layer_idx}_qk_alignment"

            # v1_Q · v1_K, v1_Q · v2_K, etc.
            for i in range(3):
                for j in range(3):
                    vals = []
                    for p in phases:
                        align = all_results[tag][p].get(key, [[0]*3]*3)
                        vals.append(align[i][j] if i < len(align) and j < len(align[i]) else 0)
                    style = '-' if i == j else '--'
                    alpha = 1.0 if i == j else 0.4
                    ax.plot(range(len(phases)), vals, f'o{style}',
                            label=f"Q_v{i+1}·K_v{j+1}", markersize=4, alpha=alpha)

            ax.set_xticks(range(len(phases)))
            ax.set_xticklabels(phases, fontsize=8)
            ax.set_ylabel("|cos(angle)|")
            ax.set_title(f"{tag} layer {layer_idx}")
            ax.set_ylim(-0.05, 1.05)
            if row == 0 and col == 0:
                ax.legend(fontsize=5, ncol=3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "qk_alignment_across_phases.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'qk_alignment_across_phases.png'}")

    # ── Plot 5: Dominant Fourier frequency per direction ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dominant Fourier ω in Each SVD Direction", fontsize=14)

    for row, tag in enumerate(["grok", "memo"]):
        for col, (layer_idx, wname) in enumerate([(0, "WQ"), (1, "WQ")]):
            ax = axes[row, col]
            phases = list(phase_steps.keys())
            key = f"layer_{layer_idx}_{wname}"

            for k in range(5):
                vals = []
                for p in phases:
                    omegas = all_results[tag][p].get(key, {}).get("dominant_omegas", [0]*5)
                    vals.append(omegas[k] if k < len(omegas) else 0)
                ax.plot(range(len(phases)), vals, 'o-', label=f"v{k+1}", markersize=5)

            ax.set_xticks(range(len(phases)))
            ax.set_xticklabels(phases, fontsize=8)
            ax.set_ylabel("Dominant ω")
            ax.set_title(f"{tag} L{layer_idx} {wname}")
            ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fourier_per_svd_direction.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'fourier_per_svd_direction.png'}")

    # Save results
    torch.save(all_results, FIG_DIR / "principal_direction_results.pt")
    print(f"Saved: {FIG_DIR / 'principal_direction_results.pt'}")

    # ── Print summary ──
    print("\n" + "="*70)
    print("SUMMARY: Edge vs Bulk Direction Analysis")
    print("="*70)

    for tag in ["grok", "memo"]:
        print(f"\n{'─'*50}")
        print(f"  {tag.upper()}")
        print(f"{'─'*50}")
        for phase in phase_steps:
            print(f"\n  {phase}:")
            for key_short in ["layer_1_WQ", "layer_1_WK"]:
                r = all_results[tag][phase].get(key_short, {})
                sv = r.get("singular_values", [])
                gap = sv[0] - sv[1] if len(sv) >= 2 else 0
                print(f"    {key_short}: σ₁={sv[0]:.3f} σ₂={sv[1]:.3f} gap={gap:.3f}"
                      if len(sv) >= 2 else f"    {key_short}: N/A")
                print(f"      depth R²: v1={r.get('depth_r2_v1', 0):.3f}, "
                      f"v2+v3={r.get('depth_r2_bulk', 0):.3f}, "
                      f"all5={r.get('depth_r2_all5', 0):.3f}")
                print(f"      token sep: {r.get('token_separation', [])}")
                print(f"      ω per dir: {r.get('dominant_omegas', [])}")


if __name__ == "__main__":
    main()
