#!/usr/bin/env python3
"""
Functional content of Gram matrix spectral edge directions.

Following the paper's methodology exactly:
  1. Compute update deltas δ_t = θ_t - θ_{t-1} (flattened attn params)
  2. Form rolling-window update matrix X(t) = [δ_{t-W+1}, ..., δ_t]
  3. SVD of X(t) → right singular vectors v_1, ..., v_W (spectral edge)
  4. Perturbation response: f_k(x) = ||h(x; θ+εv_k) - h(x; θ)||²
  5. Fourier analysis of f_k over input domain
  6. Depth probing of projected representations

Compares edge (v1, v2) vs bulk (v3+) pre/at/post grokking.
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

from dyck.grok_sweep import (
    DyckTransformerLM, VOCAB_SIZE, build_depth_dataset, split_dataset,
    TOK_OPEN, TOK_CLOSE,
)

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"

GRAM_WINDOW = 5  # rolling window size
EPS_SCALE = 0.005  # perturbation scale relative to ||θ_attn||


def get_attn_param_vector(state_dict):
    """Flatten all attention parameters into a single vector.
    Matches the paper: W_Q, W_K, W_V, W_O from all layers."""
    parts = []
    for key in sorted(state_dict.keys()):
        # Match attention projection weights
        if any(k in key for k in ['in_proj_weight', 'out_proj.weight']):
            parts.append(state_dict[key].reshape(-1).float())
    return torch.cat(parts)


def get_attn_param_keys(state_dict):
    """Get ordered list of (key, numel) for attention params."""
    keys = []
    for key in sorted(state_dict.keys()):
        if any(k in key for k in ['in_proj_weight', 'out_proj.weight']):
            keys.append((key, state_dict[key].numel()))
    return keys


def compute_gram_svd(snapshots, center_idx, window):
    """Compute SVD of rolling-window update matrix centered at center_idx.

    X = [δ_{t-W+1}, ..., δ_t] where δ_t = θ_t - θ_{t-1}

    Returns:
        singular_values: [W]
        V: [W, p] right singular vectors (directions in param space)
        steps: list of steps used
    """
    # Compute deltas
    start = max(1, center_idx - window + 1)
    end = min(len(snapshots) - 1, center_idx)

    if end - start + 1 < 2:
        return None

    deltas = []
    steps_used = []
    for i in range(start, end + 1):
        theta_prev = get_attn_param_vector(snapshots[i-1]["state_dict"])
        theta_curr = get_attn_param_vector(snapshots[i]["state_dict"])
        deltas.append((theta_curr - theta_prev).numpy())
        steps_used.append(snapshots[i]["step"])

    X = np.stack(deltas)  # [W_actual, p]
    U, S, Vh = np.linalg.svd(X, full_matrices=False)

    return {
        "singular_values": S,
        "Vh": Vh,  # [min(W,p), p] — right singular vectors
        "steps_used": steps_used,
        "g23": S[1]**2 - S[2]**2 if len(S) > 2 else 0,
    }


def perturb_model_along_direction(model, state_dict, direction_vec, param_keys, eps):
    """Create a perturbed model: θ + ε * v_k.

    direction_vec: [p] flattened direction in attn param space
    param_keys: [(key, numel)] ordered list
    eps: perturbation magnitude
    """
    perturbed_sd = {k: v.clone() for k, v in state_dict.items()}

    offset = 0
    for key, numel in param_keys:
        chunk = direction_vec[offset:offset + numel]
        perturbed_sd[key] = perturbed_sd[key] + eps * torch.tensor(
            chunk, dtype=perturbed_sd[key].dtype
        ).reshape(perturbed_sd[key].shape)
        offset += numel

    model_copy = type(model)(
        vocab_size=VOCAB_SIZE,
        ctx_len=model.ctx_len,
        d_model=model.head.in_features,
        n_layers=len(model.encoder.layers),
        n_heads=model.encoder.layers[0].self_attn.num_heads,
        d_ff=model.encoder.layers[0].linear1.out_features,
        dropout=0.0,
    )
    model_copy.load_state_dict(perturbed_sd)
    model_copy.eval()
    return model_copy


def compute_perturbation_response(model, state_dict, direction_vec, param_keys,
                                   test_x, eps):
    """Compute f_k(x) = ||h(x; θ+εv_k) - h(x; θ)||² for all inputs.

    Returns: [N, T] perturbation response field
    """
    model.eval()

    # Get baseline residual stream (after encoder, before head)
    with torch.no_grad():
        B, T = test_x.shape
        pos = torch.arange(T)
        h_base = model.tok_emb(test_x) + model.pos_emb(pos)[None, :, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T)
        h_base = model.encoder(h_base, mask=mask, is_causal=True)
        h_base = model.ln(h_base)  # [B, T, d_model]

    # Get perturbed residual stream
    model_perturbed = perturb_model_along_direction(
        model, state_dict, direction_vec, param_keys, eps)

    with torch.no_grad():
        h_pert = model_perturbed.tok_emb(test_x) + model_perturbed.pos_emb(pos)[None, :, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T)
        h_pert = model_perturbed.encoder(h_pert, mask=mask, is_causal=True)
        h_pert = model_perturbed.ln(h_pert)

    # f_k(x) = ||Δh||²
    delta_h = (h_pert - h_base).numpy()
    f_k = np.sum(delta_h ** 2, axis=-1)  # [N, T]

    return f_k, delta_h


def analyze_perturbation_field(f_k, delta_h, depths, tokens, mask):
    """Analyze the perturbation response field f_k(x).

    Returns: Fourier concentration, depth correlation, token sensitivity
    """
    N, T = f_k.shape
    mask_bool = mask.astype(bool)

    # 1. Depth-conditioned mean perturbation
    depth_means = {}
    for d in range(13):
        idx = (depths == d) & mask_bool
        if idx.sum() > 0:
            depth_means[d] = f_k[idx].mean()

    # Depth correlation
    f_valid = f_k[mask_bool]
    d_valid = depths[mask_bool]
    if len(np.unique(d_valid)) > 1:
        from scipy.stats import pearsonr
        depth_corr, _ = pearsonr(f_valid, d_valid)
    else:
        depth_corr = 0.0

    # 2. Token sensitivity: open vs close
    open_mask = (tokens == TOK_OPEN) & mask_bool
    close_mask = (tokens == TOK_CLOSE) & mask_bool
    f_open = f_k[open_mask].mean() if open_mask.sum() > 0 else 0
    f_close = f_k[close_mask].mean() if close_mask.sum() > 0 else 0
    token_ratio = f_open / (f_close + 1e-10)

    # 3. Fourier analysis of f_k along position
    # Average f_k across sequences, then DFT
    f_mean_pos = np.zeros(T)
    for t in range(T):
        valid = mask_bool[:, t]
        if valid.sum() > 0:
            f_mean_pos[t] = f_k[valid, t].mean()

    fft = np.fft.rfft(f_mean_pos)
    power = np.abs(fft) ** 2
    total_power = power.sum()
    if total_power > 0:
        fourier_conc = power.max() / total_power
        dominant_omega = np.argmax(power)
    else:
        fourier_conc = 0
        dominant_omega = 0

    # 4. Depth probe R² using delta_h
    D = delta_h.shape[-1]
    dh_flat = delta_h.reshape(-1, D)[mask_bool.reshape(-1)]
    d_flat = depths.reshape(-1)[mask_bool.reshape(-1)]
    n = len(dh_flat)
    perm = np.random.permutation(n)
    s = int(0.7 * n)
    probe = Ridge(alpha=1.0)
    probe.fit(dh_flat[perm[:s]], d_flat[perm[:s]])
    pred = probe.predict(dh_flat[perm[s:]])
    depth_r2 = r2_score(d_flat[perm[s:]], pred)

    return {
        "depth_corr": depth_corr,
        "depth_r2": depth_r2,
        "f_open": float(f_open),
        "f_close": float(f_close),
        "token_ratio": float(token_ratio),
        "fourier_conc": float(fourier_conc),
        "dominant_omega": int(dominant_omega),
        "depth_means": depth_means,
        "f_mean_pos": f_mean_pos,
        "power_spectrum": power,
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # Load test data
    X_all, Y_all = build_depth_dataset(n_seqs=5050, max_pairs=12, ctx_len=24, seed=0)
    _, _, test_x, test_y = split_dataset(X_all, Y_all, frac_train=50/5050, seed=0)
    test_x = test_x[:300]
    test_y = test_y[:300]
    mask = (test_y != -100).numpy()
    depths = test_y.numpy()
    tokens = test_x.numpy()

    # Analysis phases (checkpoint indices to center the Gram window)
    # We need at least GRAM_WINDOW consecutive snapshots before the center
    phase_indices = {
        "pre_grok": 3,   # ~step 300
        "at_grok": 6,    # ~step 600
        "post_grok": 15, # ~step 1500
        "late": 40,      # ~step 4000
    }

    all_results = {}

    for tag, ckpt_name in [("grok", "dyck_grok_fourier.pt"), ("memo", "dyck_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        ckpt = torch.load(ckpt_path, weights_only=False)
        snapshots = ckpt["snapshots"]
        cfg = ckpt["cfg"]

        print(f"\n{'='*60}")
        print(f"Gram edge functional modes: {tag}")
        print(f"  {len(snapshots)} snapshots available")
        print(f"{'='*60}")

        # Build model for perturbation
        model = DyckTransformerLM(
            vocab_size=VOCAB_SIZE,
            ctx_len=max(cfg["CTX_LEN"], cfg["CTX_LEN_OOD"]),
            d_model=cfg["D_MODEL"], n_layers=cfg["N_LAYERS"],
            n_heads=cfg["N_HEADS"], d_ff=cfg["D_FF"],
            dropout=cfg["DROPOUT"], n_classes=cfg["N_CLASSES"],
        )

        param_keys = get_attn_param_keys(snapshots[0]["state_dict"])
        p = sum(n for _, n in param_keys)
        print(f"  Attn param dim p = {p}")

        all_results[tag] = {}

        for phase_name, center_idx in phase_indices.items():
            if center_idx >= len(snapshots):
                print(f"\n  {phase_name}: SKIP (not enough snapshots)")
                continue

            center_step = snapshots[center_idx]["step"]
            print(f"\n  {phase_name} (center step {center_step}, idx {center_idx}):")

            # Compute Gram SVD
            gram = compute_gram_svd(snapshots, center_idx, GRAM_WINDOW)
            if gram is None:
                print("    Not enough deltas for Gram")
                continue

            S = gram["singular_values"]
            Vh = gram["Vh"]
            print(f"    σ = {S[:6].round(4)}")
            print(f"    g₂₃ = {gram['g23']:.6f}")

            # Load model at this step
            model.load_state_dict(snapshots[center_idx]["state_dict"])
            model.eval()

            # Compute eps
            theta_attn = get_attn_param_vector(snapshots[center_idx]["state_dict"])
            eps = EPS_SCALE * torch.norm(theta_attn).item()
            print(f"    ε = {eps:.4f}")

            phase_results = {
                "step": center_step,
                "singular_values": S.tolist(),
                "g23": gram["g23"],
                "directions": {},
            }

            # Analyze each direction
            n_dirs = min(len(S), 6)
            for k in range(n_dirs):
                v_k = Vh[k]  # [p] direction in attn param space

                f_k, delta_h = compute_perturbation_response(
                    model, snapshots[center_idx]["state_dict"],
                    v_k, param_keys, test_x, eps
                )

                analysis = analyze_perturbation_field(f_k, delta_h, depths, tokens, mask)

                label = "EDGE" if k < 2 else "BULK"
                print(f"    v{k+1} ({label}): σ={S[k]:.4f}, "
                      f"depth_R²={analysis['depth_r2']:.3f}, "
                      f"depth_corr={analysis['depth_corr']:.3f}, "
                      f"tok_ratio={analysis['token_ratio']:.2f}, "
                      f"ω*={analysis['dominant_omega']}, "
                      f"F_conc={analysis['fourier_conc']:.3f}")

                phase_results["directions"][f"v{k+1}"] = {
                    "sigma": float(S[k]),
                    "depth_r2": analysis["depth_r2"],
                    "depth_corr": analysis["depth_corr"],
                    "token_ratio": analysis["token_ratio"],
                    "fourier_conc": analysis["fourier_conc"],
                    "dominant_omega": analysis["dominant_omega"],
                    "f_open": analysis["f_open"],
                    "f_close": analysis["f_close"],
                }

            all_results[tag][phase_name] = phase_results

    # ══════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════

    phases = list(phase_indices.keys())

    # ── Plot 1: Singular values and g23 across phases ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Gram Matrix Spectral Edge Across Grokking Phases", fontsize=14)

    for i, tag in enumerate(["grok", "memo"]):
        ax = axes[i]
        for k in range(min(5, 6)):
            vals = []
            for p in phases:
                r = all_results.get(tag, {}).get(p, {})
                sv = r.get("singular_values", [0]*6)
                vals.append(sv[k] if k < len(sv) else 0)
            ax.plot(range(len(phases)), vals, 'o-', label=f"σ_{k+1}", markersize=5)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=8)
        ax.set_title(f"{tag}: Update Gram singular values")
        ax.set_ylabel("σ")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "gram_sv_across_phases.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Edge vs Bulk depth R² ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Depth R² of Perturbation Response: Edge vs Bulk", fontsize=14)

    for i, tag in enumerate(["grok", "memo"]):
        ax = axes[i]
        for k, label, color in [(0, "v1 (edge)", "steelblue"), (1, "v2 (edge)", "royalblue"),
                                 (2, "v3 (bulk)", "coral"), (3, "v4 (bulk)", "lightsalmon")]:
            vals = []
            for p in phases:
                r = all_results.get(tag, {}).get(p, {})
                d = r.get("directions", {}).get(f"v{k+1}", {})
                vals.append(d.get("depth_r2", 0))
            ax.plot(range(len(phases)), vals, 'o-', color=color, label=label, markersize=6)

        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=8)
        ax.set_ylabel("R² (depth from Δh)")
        ax.set_title(f"{tag}: Perturbation → Depth")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "gram_edge_vs_bulk_depth.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Token sensitivity (open/close ratio) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Token Sensitivity Ratio (f_open / f_close) per Direction", fontsize=14)

    for i, tag in enumerate(["grok", "memo"]):
        ax = axes[i]
        for k, label, color in [(0, "v1", "steelblue"), (1, "v2", "royalblue"),
                                 (2, "v3", "coral"), (3, "v4", "lightsalmon")]:
            vals = []
            for p in phases:
                r = all_results.get(tag, {}).get(p, {})
                d = r.get("directions", {}).get(f"v{k+1}", {})
                vals.append(d.get("token_ratio", 1.0))
            ax.plot(range(len(phases)), vals, 'o-', color=color, label=label, markersize=6)

        ax.axhline(y=1.0, ls='--', color='gray', alpha=0.5, label='equal sensitivity')
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=8)
        ax.set_ylabel("f(open) / f(close)")
        ax.set_title(f"{tag}: Token sensitivity ratio")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "gram_token_sensitivity.png", dpi=150)
    plt.close(fig)

    # ── Plot 4: Fourier concentration per direction ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fourier Concentration of Perturbation Response", fontsize=14)

    for i, tag in enumerate(["grok", "memo"]):
        ax = axes[i]
        for k, label, color in [(0, "v1 (edge)", "steelblue"), (1, "v2 (edge)", "royalblue"),
                                 (2, "v3 (bulk)", "coral"), (3, "v4 (bulk)", "lightsalmon")]:
            vals = []
            omegas = []
            for p in phases:
                r = all_results.get(tag, {}).get(p, {})
                d = r.get("directions", {}).get(f"v{k+1}", {})
                vals.append(d.get("fourier_conc", 0))
                omegas.append(d.get("dominant_omega", 0))
            ax.plot(range(len(phases)), vals, 'o-', color=color, label=label, markersize=6)
            # Annotate with dominant omega
            for j, (v, w) in enumerate(zip(vals, omegas)):
                ax.annotate(f"ω={w}", (j, v), fontsize=6, ha='center', va='bottom')

        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=8)
        ax.set_ylabel("Fourier concentration F_k")
        ax.set_title(f"{tag}: f_k Fourier structure")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "gram_fourier_concentration.png", dpi=150)
    plt.close(fig)

    # Save results
    torch.save(all_results, FIG_DIR / "gram_edge_functional_results.pt")

    print(f"\nSaved all figures to {FIG_DIR}")
    print(f"Saved results to {FIG_DIR / 'gram_edge_functional_results.pt'}")

    # ── Final summary ──
    print("\n" + "="*70)
    print("SUMMARY: Gram Edge vs Bulk Functional Modes")
    print("="*70)

    for tag in ["grok", "memo"]:
        print(f"\n{'─'*50}")
        print(f"  {tag.upper()}")
        print(f"{'─'*50}")
        for phase in phases:
            r = all_results.get(tag, {}).get(phase, {})
            print(f"\n  {phase} (step {r.get('step', '?')}):")
            print(f"    g₂₃ = {r.get('g23', 0):.6f}")
            for vname in ["v1", "v2", "v3", "v4"]:
                d = r.get("directions", {}).get(vname, {})
                if d:
                    label = "EDGE" if vname in ["v1", "v2"] else "BULK"
                    print(f"    {vname} ({label}): σ={d.get('sigma', 0):.4f}, "
                          f"depth_R²={d.get('depth_r2', 0):.3f}, "
                          f"tok_ratio={d.get('token_ratio', 0):.2f}, "
                          f"ω*={d.get('dominant_omega', 0)}, "
                          f"F={d.get('fourier_conc', 0):.3f}")


if __name__ == "__main__":
    main()
