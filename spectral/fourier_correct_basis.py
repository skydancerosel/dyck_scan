#!/usr/bin/env python3
"""
#1: Fourier analysis of Gram edge directions in the CORRECT basis.

For modular arithmetic, the paper groups by (a+b) mod p and finds F=0.40.
For Dyck, the correct grouping variable is DEPTH (0..12).
For SCAN, the correct grouping is ACTION TYPE.

For each Gram direction v_k:
  - Compute f_k(x) = ||Δh_k(x)||² (perturbation response)
  - Group by depth/action → f̄_k[d] = mean f_k over positions at depth d
  - DFT of f̄_k → Fourier concentration F_k
  - Compare edge vs bulk, grok vs memo

#2: Per-block decomposition of Gram directions.
  - Project v_k into W_Q / W_K / W_V / W_O blocks
  - Compute per-block perturbation response
  - Fourier analysis of each block's contribution
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

from dyck.grok_sweep import (
    DyckTransformerLM, VOCAB_SIZE, build_depth_dataset, split_dataset,
    TOK_OPEN, TOK_CLOSE,
)
from spectral.gram_edge_functional_modes import (
    get_attn_param_vector, get_attn_param_keys, compute_gram_svd,
)

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"
GRAM_WINDOW = 5
EPS_SCALE = 0.005


def build_model(cfg):
    return DyckTransformerLM(
        vocab_size=VOCAB_SIZE,
        ctx_len=max(cfg["CTX_LEN"], cfg["CTX_LEN_OOD"]),
        d_model=cfg["D_MODEL"], n_layers=cfg["N_LAYERS"],
        n_heads=cfg["N_HEADS"], d_ff=cfg["D_FF"],
        dropout=cfg["DROPOUT"], n_classes=cfg["N_CLASSES"],
    )


def get_block_ranges(state_dict):
    """Get (key, offset, numel) for each attention parameter block."""
    blocks = []
    offset = 0
    for key in sorted(state_dict.keys()):
        if any(k in key for k in ['in_proj_weight', 'out_proj.weight']):
            numel = state_dict[key].numel()
            # Identify block type
            if 'in_proj_weight' in key:
                d = state_dict[key].shape[1]
                # in_proj_weight is [3*d, d] = [WQ; WK; WV]
                blocks.append({"key": key, "type": "WQ", "offset": offset, "numel": d*d,
                               "layer": key.split('.')[2]})
                blocks.append({"key": key, "type": "WK", "offset": offset + d*d, "numel": d*d,
                               "layer": key.split('.')[2]})
                blocks.append({"key": key, "type": "WV", "offset": offset + 2*d*d, "numel": d*d,
                               "layer": key.split('.')[2]})
            elif 'out_proj.weight' in key:
                blocks.append({"key": key, "type": "WO", "offset": offset, "numel": numel,
                               "layer": key.split('.')[2]})
            offset += numel
    return blocks


def compute_perturbation_response(model, state_dict, cfg, direction_vec, param_keys,
                                   test_x, eps):
    """Compute f_k(x) = ||Δh_k(x)||² and return per-position values + delta_h."""
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        B, T = test_x.shape
        pos = torch.arange(T)
        h_base = model.tok_emb(test_x) + model.pos_emb(pos)[None, :, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T)
        h_base = model.encoder(h_base, mask=mask, is_causal=True)
        h_base = model.ln(h_base)

    # Perturb
    perturbed_sd = {k: v.clone() for k, v in state_dict.items()}
    offset = 0
    for key, numel in param_keys:
        chunk = direction_vec[offset:offset + numel]
        perturbed_sd[key] = perturbed_sd[key] + eps * torch.tensor(
            chunk, dtype=perturbed_sd[key].dtype).reshape(perturbed_sd[key].shape)
        offset += numel

    model_p = build_model(cfg)
    model_p.load_state_dict(perturbed_sd)
    model_p.eval()

    with torch.no_grad():
        h_pert = model_p.tok_emb(test_x) + model_p.pos_emb(pos)[None, :, :]
        h_pert = model_p.encoder(h_pert, mask=mask, is_causal=True)
        h_pert = model_p.ln(h_pert)

    delta_h = (h_pert - h_base).numpy()
    f_k = np.sum(delta_h ** 2, axis=-1)  # [N, T]
    return f_k, delta_h


def compute_block_perturbation(model, state_dict, cfg, direction_vec, param_keys,
                                blocks, block_type, test_x, eps):
    """Perturb ONLY the specified block type (WQ, WK, WV, WO) and measure response."""
    # Zero out all blocks except the target type
    masked_direction = np.zeros_like(direction_vec)
    for b in blocks:
        if b["type"] == block_type:
            masked_direction[b["offset"]:b["offset"] + b["numel"]] = \
                direction_vec[b["offset"]:b["offset"] + b["numel"]]

    return compute_perturbation_response(model, state_dict, cfg, masked_direction,
                                          param_keys, test_x, eps)


def depth_grouped_fourier(f_k, depths, mask):
    """Group f_k by depth, compute DFT, return Fourier concentration.

    f̄[d] = mean f_k over all (sequence, position) where depth = d
    Then DFT of f̄ → F_k = max|F̂[ω]|² / Σ|F̂[ω]|²
    """
    mask_bool = mask.astype(bool)
    unique_depths = sorted(set(depths[mask_bool].tolist()))

    if len(unique_depths) < 3:
        return {"concentration": 0, "dominant_omega": 0, "signal": [], "depths": []}

    # Build signal: mean perturbation at each depth
    signal = []
    for d in unique_depths:
        idx = (depths == d) & mask_bool
        if idx.sum() > 0:
            signal.append(f_k[idx].mean())
        else:
            signal.append(0)
    signal = np.array(signal)

    # Remove DC component (mean) to focus on variation
    signal_centered = signal - signal.mean()

    # DFT
    fft = np.fft.rfft(signal_centered)
    power = np.abs(fft) ** 2
    total = power.sum()

    if total > 0:
        concentration = power.max() / total
        dominant_omega = int(np.argmax(power))
    else:
        concentration = 0
        dominant_omega = 0

    # Uniform baseline: 1 / (len(signal)//2)
    n_freqs = len(power)
    uniform_baseline = 1.0 / n_freqs if n_freqs > 0 else 0

    return {
        "concentration": float(concentration),
        "dominant_omega": dominant_omega,
        "signal": signal.tolist(),
        "depths": unique_depths,
        "power": power.tolist(),
        "uniform_baseline": uniform_baseline,
        "elevation": float(concentration / uniform_baseline) if uniform_baseline > 0 else 0,
    }


def token_grouped_analysis(f_k, tokens, mask):
    """Group f_k by token type (open vs close)."""
    mask_bool = mask.astype(bool)
    f_open = f_k[(tokens == TOK_OPEN) & mask_bool].mean() if ((tokens == TOK_OPEN) & mask_bool).sum() > 0 else 0
    f_close = f_k[(tokens == TOK_CLOSE) & mask_bool].mean() if ((tokens == TOK_CLOSE) & mask_bool).sum() > 0 else 0
    return {"f_open": float(f_open), "f_close": float(f_close),
            "ratio": float(f_open / (f_close + 1e-12))}


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # Test data
    X_all, Y_all = build_depth_dataset(n_seqs=5050, max_pairs=12, ctx_len=24, seed=0)
    _, _, test_x, test_y = split_dataset(X_all, Y_all, frac_train=50/5050, seed=0)
    test_x = test_x[:300]
    test_y = test_y[:300]
    mask = (test_y != -100).numpy()
    depths = test_y.numpy()
    tokens = test_x.numpy()

    phase_indices = {"pre_grok": 2, "at_grok": 5, "post_grok": 14, "late": 39}
    phases = list(phase_indices.keys())

    all_results = {}

    for tag, ckpt_name in [("grok", "dyck_grok_fourier.pt"), ("memo", "dyck_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        ckpt = torch.load(ckpt_path, weights_only=False)
        snapshots = ckpt["snapshots"]
        cfg = ckpt["cfg"]
        param_keys = get_attn_param_keys(snapshots[0]["state_dict"])
        blocks = get_block_ranges(snapshots[0]["state_dict"])

        model = build_model(cfg)

        print(f"\n{'='*60}")
        print(f"Fourier correct basis + per-block: {tag}")
        print(f"{'='*60}")

        all_results[tag] = {}

        for phase_name, center_idx in phase_indices.items():
            if center_idx >= len(snapshots):
                continue

            state_dict = snapshots[center_idx]["state_dict"]
            step = snapshots[center_idx]["step"]
            gram = compute_gram_svd(snapshots, center_idx, GRAM_WINDOW)
            if gram is None:
                continue

            S = gram["singular_values"]
            Vh = gram["Vh"]
            theta_attn = get_attn_param_vector(state_dict)
            eps = EPS_SCALE * torch.norm(theta_attn).item()

            print(f"\n  {phase_name} (step {step}):")

            phase_results = {"step": step, "directions": {}}

            n_dirs = min(4, len(S))
            for k in range(n_dirs):
                v_k = Vh[k]
                label = "EDGE" if k < 2 else "BULK"

                # Full perturbation response
                f_k, delta_h = compute_perturbation_response(
                    model, state_dict, cfg, v_k, param_keys, test_x, eps)

                # Depth-grouped Fourier
                depth_fourier = depth_grouped_fourier(f_k, depths, mask)
                token_analysis = token_grouped_analysis(f_k, tokens, mask)

                print(f"    v{k+1} ({label}): depth F={depth_fourier['concentration']:.3f} "
                      f"({depth_fourier['elevation']:.1f}x uniform), "
                      f"ω*={depth_fourier['dominant_omega']}, "
                      f"tok_ratio={token_analysis['ratio']:.2f}")

                dir_result = {
                    "sigma": float(S[k]),
                    "depth_fourier": depth_fourier,
                    "token": token_analysis,
                    "per_block": {},
                }

                # Per-block decomposition
                for block_type in ["WQ", "WK", "WV", "WO"]:
                    f_block, _ = compute_block_perturbation(
                        model, state_dict, cfg, v_k, param_keys, blocks,
                        block_type, test_x, eps)
                    block_fourier = depth_grouped_fourier(f_block, depths, mask)
                    block_token = token_grouped_analysis(f_block, tokens, mask)

                    dir_result["per_block"][block_type] = {
                        "depth_fourier": block_fourier,
                        "token": block_token,
                    }

                    print(f"      {block_type}: depth F={block_fourier['concentration']:.3f} "
                          f"({block_fourier['elevation']:.1f}x), "
                          f"tok={block_token['ratio']:.2f}")

                phase_results["directions"][f"v{k+1}"] = dir_result

            all_results[tag][phase_name] = phase_results

    # ══════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════

    # ── Plot 1: Depth-grouped Fourier concentration (edge vs bulk, across phases) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fourier Concentration in Depth Basis (Correct Basis for Dyck)", fontsize=14)

    for i, tag in enumerate(["grok", "memo"]):
        ax = axes[i]
        for k, color in [(0, "steelblue"), (1, "royalblue"), (2, "coral"), (3, "lightsalmon")]:
            vals = []
            for p in phases:
                r = all_results[tag].get(p, {}).get("directions", {}).get(f"v{k+1}", {})
                df = r.get("depth_fourier", {})
                vals.append(df.get("elevation", 0))
            label = "edge" if k < 2 else "bulk"
            ax.plot(range(len(phases)), vals, 'o-', color=color,
                    label=f"v{k+1} ({label})", markersize=6)
        ax.axhline(y=1, ls='--', color='gray', alpha=0.5, label='uniform')
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=8)
        ax.set_ylabel("Fourier concentration / uniform baseline")
        ax.set_title(f"{tag}")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fourier_depth_basis_concentration.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Per-block Fourier analysis ──
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Per-Block Fourier Concentration (Depth Basis): Grok Late", fontsize=14)

    for row, tag in enumerate(["grok", "memo"]):
        r = all_results[tag].get("late", {})
        for col, block_type in enumerate(["WQ", "WK", "WV", "WO"]):
            ax = axes[row, col]
            for k in range(min(4, len(r.get("directions", {})))):
                d = r.get("directions", {}).get(f"v{k+1}", {})
                block_data = d.get("per_block", {}).get(block_type, {})
                df = block_data.get("depth_fourier", {})
                elev = df.get("elevation", 0)
                color = "steelblue" if k < 2 else "coral"
                ax.bar(k, elev, color=color, alpha=0.8)
            ax.axhline(y=1, ls='--', color='gray', alpha=0.5)
            ax.set_title(f"{tag} {block_type}", fontsize=10)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["v1", "v2", "v3", "v4"], fontsize=8)
            if col == 0:
                ax.set_ylabel("Elevation over uniform")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "per_block_fourier_depth.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Depth-conditioned perturbation signal (the f̄[d] curves) ──
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Perturbation Response Grouped by Depth: f̄_k[depth]", fontsize=14)

    for row, tag in enumerate(["grok", "memo"]):
        for col, phase in enumerate(phases):
            ax = axes[row, col]
            r = all_results[tag].get(phase, {})
            for k in range(min(4, len(r.get("directions", {})))):
                d = r.get("directions", {}).get(f"v{k+1}", {})
                df = d.get("depth_fourier", {})
                signal = df.get("signal", [])
                dep = df.get("depths", [])
                if signal and dep:
                    color = "steelblue" if k < 2 else "coral"
                    label_type = "edge" if k < 2 else "bulk"
                    # Normalize
                    s = np.array(signal)
                    if s.max() > 0:
                        s = s / s.max()
                    ax.plot(dep, s, 'o-', color=color, label=f"v{k+1} ({label_type})",
                            markersize=4, linewidth=1.5)
            ax.set_title(f"{tag} {phase}", fontsize=9)
            ax.set_xlabel("Depth")
            if col == 0:
                ax.set_ylabel("f̄[d] (normalized)")
            if row == 0 and col == 0:
                ax.legend(fontsize=6)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "perturbation_by_depth.png", dpi=150)
    plt.close(fig)

    # Save
    torch.save(all_results, FIG_DIR / "fourier_correct_basis_results.pt")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Fourier Analysis in Correct Basis (Depth)")
    print("="*70)
    for tag in ["grok", "memo"]:
        print(f"\n  {tag}:")
        for phase in phases:
            r = all_results[tag].get(phase, {})
            if not r:
                continue
            print(f"    {phase} (step {r.get('step', '?')}):")
            for vname in ["v1", "v2", "v3", "v4"]:
                d = r.get("directions", {}).get(vname, {})
                if d:
                    df = d.get("depth_fourier", {})
                    label = "EDGE" if vname in ["v1", "v2"] else "BULK"
                    print(f"      {vname} ({label}): F={df.get('concentration', 0):.3f} "
                          f"({df.get('elevation', 0):.1f}x), ω*={df.get('dominant_omega', 0)}")
                    # Per-block summary
                    for bt in ["WQ", "WK"]:
                        bd = d.get("per_block", {}).get(bt, {}).get("depth_fourier", {})
                        print(f"        {bt}: F={bd.get('concentration', 0):.3f} "
                              f"({bd.get('elevation', 0):.1f}x)")


if __name__ == "__main__":
    main()
