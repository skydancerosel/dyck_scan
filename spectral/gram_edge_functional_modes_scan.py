#!/usr/bin/env python3
"""
Functional content of Gram matrix spectral edge directions — SCAN.

Same methodology as gram_edge_functional_modes.py but for the
encoder-decoder SCAN model:
  1. δ_t = θ_t - θ_{t-1} (flattened attn params from enc+dec)
  2. Rolling-window Gram SVD → v1, ..., v_W
  3. Perturbation response: f_k(x) = ||h(x; θ+εv_k) - h(x; θ)||²
  4. Probe for action semantics, Fourier structure, command sensitivity
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

from scan.grok_sweep import ScanTransformer

CKPT_DIR = Path(__file__).resolve().parent / "fourier_scan_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_scan_plots"

GRAM_WINDOW = 5
EPS_SCALE = 0.005


def get_attn_param_vector(state_dict):
    """Flatten all attention parameters (enc + dec self + dec cross) into one vector."""
    parts = []
    for key in sorted(state_dict.keys()):
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
    """Compute SVD of rolling-window update matrix."""
    start = max(1, center_idx - window + 1)
    end = min(len(snapshots) - 1, center_idx)
    if end - start + 1 < 2:
        return None

    deltas = []
    for i in range(start, end + 1):
        theta_prev = get_attn_param_vector(snapshots[i-1]["state_dict"])
        theta_curr = get_attn_param_vector(snapshots[i]["state_dict"])
        deltas.append((theta_curr - theta_prev).numpy())

    X = np.stack(deltas)
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    return {
        "singular_values": S,
        "Vh": Vh,
        "g23": S[1]**2 - S[2]**2 if len(S) > 2 else 0,
    }


def build_scan_model(cfg, ckpt):
    """Instantiate a SCAN model from checkpoint metadata."""
    return ScanTransformer(
        src_vocab_size=ckpt["cmd_vocab"].size,
        tgt_vocab_size=ckpt["act_vocab"].size,
        max_src_len=ckpt["max_cmd_len"],
        max_tgt_len=ckpt["max_act_len"],
        d_model=cfg["D_MODEL"], n_layers=cfg["N_LAYERS"],
        n_heads=cfg["N_HEADS"], d_ff=cfg["D_FF"],
        dropout=cfg["DROPOUT"],
    )


def compute_perturbation_response_scan(model, state_dict, direction_vec, param_keys,
                                        src, tgt_in, eps, cfg, ckpt):
    """Compute f_k for encoder output and decoder output."""
    pad_id = 0
    model.eval()

    # Baseline forward
    with torch.no_grad():
        src_pad = (src == pad_id)
        tgt_pad = (tgt_in == pad_id)

        B, S = src.shape
        _, T = tgt_in.shape

        # Encoder output
        src_pos = torch.arange(S)
        src_emb = model.src_tok_emb(src) + model.src_pos_emb(src_pos)[None, :, :]
        memory_base = model.transformer.encoder(src_emb, src_key_padding_mask=src_pad)

        # Decoder output
        tgt_pos = torch.arange(T)
        tgt_emb = model.tgt_tok_emb(tgt_in) + model.tgt_pos_emb(tgt_pos)[None, :, :]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T)
        dec_base = model.transformer.decoder(
            tgt_emb, memory_base,
            tgt_mask=tgt_mask, tgt_is_causal=True,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        dec_base = model.ln(dec_base)

    # Perturbed model
    perturbed_sd = {k: v.clone() for k, v in state_dict.items()}
    offset = 0
    for key, numel in param_keys:
        chunk = direction_vec[offset:offset + numel]
        perturbed_sd[key] = perturbed_sd[key] + eps * torch.tensor(
            chunk, dtype=perturbed_sd[key].dtype
        ).reshape(perturbed_sd[key].shape)
        offset += numel

    model_pert = build_scan_model(cfg, ckpt)
    model_pert.load_state_dict(perturbed_sd)
    model_pert.eval()

    with torch.no_grad():
        src_emb_p = model_pert.src_tok_emb(src) + model_pert.src_pos_emb(src_pos)[None, :, :]
        memory_pert = model_pert.transformer.encoder(src_emb_p, src_key_padding_mask=src_pad)

        tgt_emb_p = model_pert.tgt_tok_emb(tgt_in) + model_pert.tgt_pos_emb(tgt_pos)[None, :, :]
        dec_pert = model_pert.transformer.decoder(
            tgt_emb_p, memory_pert,
            tgt_mask=tgt_mask, tgt_is_causal=True,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        dec_pert = model_pert.ln(dec_pert)

    delta_enc = (memory_pert - memory_base).numpy()
    delta_dec = (dec_pert - dec_base).numpy()

    f_enc = np.sum(delta_enc ** 2, axis=-1)  # [N, S]
    f_dec = np.sum(delta_dec ** 2, axis=-1)  # [N, T]

    return f_enc, f_dec, delta_enc, delta_dec


def analyze_field(f_k, delta_h, targets, mask):
    """Analyze perturbation field: depth/token correlation, Fourier, R²."""
    N, T = f_k.shape
    mask_bool = mask.astype(bool)

    # R² from delta_h → target
    D = delta_h.shape[-1]
    dh_flat = delta_h.reshape(-1, D)[mask_bool.reshape(-1)]
    t_flat = targets.reshape(-1)[mask_bool.reshape(-1)]

    depth_r2 = 0.0
    if len(np.unique(t_flat)) > 1 and len(dh_flat) > 50:
        n = len(dh_flat)
        perm = np.random.permutation(n)
        s = int(0.7 * n)
        probe = Ridge(alpha=1.0)
        try:
            probe.fit(dh_flat[perm[:s]], t_flat[perm[:s]])
            pred = probe.predict(dh_flat[perm[s:]])
            depth_r2 = r2_score(t_flat[perm[s:]], pred)
        except Exception:
            pass

    # Fourier of position-averaged f_k
    f_mean = np.zeros(T)
    for t in range(T):
        valid = mask_bool[:, t]
        if valid.sum() > 0:
            f_mean[t] = f_k[valid, t].mean()

    fft = np.fft.rfft(f_mean)
    power = np.abs(fft) ** 2
    total = power.sum()
    fourier_conc = power.max() / total if total > 0 else 0
    dominant_omega = int(np.argmax(power))

    # Mean perturbation magnitude
    mean_f = f_k[mask_bool].mean() if mask_bool.sum() > 0 else 0

    return {
        "target_r2": depth_r2,
        "fourier_conc": float(fourier_conc),
        "dominant_omega": dominant_omega,
        "mean_f": float(mean_f),
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    phase_indices = {
        "early": 2,
        "pre_grok": 5,
        "at_grok": 10,
        "post_grok": 20,
    }

    all_results = {}

    for tag, ckpt_name in [("grok", "scan_grok_fourier.pt"), ("memo", "scan_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"Skipping {tag}")
            continue

        ckpt = torch.load(ckpt_path, weights_only=False)
        snapshots = ckpt["snapshots"]
        cfg = ckpt["cfg"]

        test_src = ckpt["test_src"][:150]
        test_tgt_in = ckpt["test_tgt_in"][:150]
        test_tgt_out = ckpt["test_tgt_out"][:150]

        src_mask = (test_src != 0).numpy()
        tgt_mask = (test_tgt_out != -100).numpy()
        tgt_tokens = test_tgt_out.numpy()

        # First action token broadcast to encoder positions (command semantics target)
        first_action = np.zeros_like(test_src.numpy())
        for i in range(len(test_tgt_out)):
            valid = test_tgt_out[i][test_tgt_out[i] != -100]
            if len(valid) > 0:
                first_action[i, :] = valid[0].item()

        print(f"\n{'='*60}")
        print(f"Gram edge functional modes: SCAN {tag}")
        print(f"  {len(snapshots)} snapshots")
        print(f"{'='*60}")

        param_keys = get_attn_param_keys(snapshots[0]["state_dict"])
        p = sum(n for _, n in param_keys)
        print(f"  Attn param dim p = {p}")

        # Build model template
        model = build_scan_model(cfg, ckpt)

        all_results[tag] = {}

        for phase_name, center_idx in phase_indices.items():
            if center_idx >= len(snapshots):
                print(f"\n  {phase_name}: SKIP")
                continue

            center_step = snapshots[center_idx]["step"]
            print(f"\n  {phase_name} (step {center_step}, idx {center_idx}):")

            gram = compute_gram_svd(snapshots, center_idx, GRAM_WINDOW)
            if gram is None:
                print("    Not enough deltas")
                continue

            S = gram["singular_values"]
            Vh = gram["Vh"]
            print(f"    σ = {S[:6].round(4)}")
            print(f"    g₂₃ = {gram['g23']:.6f}")

            # Load model at this step
            model.load_state_dict(snapshots[center_idx]["state_dict"])
            model.eval()

            theta_attn = get_attn_param_vector(snapshots[center_idx]["state_dict"])
            eps = EPS_SCALE * torch.norm(theta_attn).item()

            phase_results = {
                "step": center_step,
                "singular_values": S.tolist(),
                "g23": gram["g23"],
                "directions": {},
            }

            n_dirs = min(len(S), 5)
            for k in range(n_dirs):
                v_k = Vh[k]

                f_enc, f_dec, dh_enc, dh_dec = compute_perturbation_response_scan(
                    model, snapshots[center_idx]["state_dict"],
                    v_k, param_keys, test_src, test_tgt_in, eps, cfg, ckpt
                )

                enc_analysis = analyze_field(f_enc, dh_enc, first_action, src_mask)
                dec_analysis = analyze_field(f_dec, dh_dec, tgt_tokens, tgt_mask)

                label = "EDGE" if k < 2 else "BULK"
                print(f"    v{k+1} ({label}): σ={S[k]:.4f}")
                print(f"      enc: R²(cmd→action)={enc_analysis['target_r2']:.3f}, "
                      f"ω*={enc_analysis['dominant_omega']}, F={enc_analysis['fourier_conc']:.3f}")
                print(f"      dec: R²(→token)={dec_analysis['target_r2']:.3f}, "
                      f"ω*={dec_analysis['dominant_omega']}, F={dec_analysis['fourier_conc']:.3f}")

                phase_results["directions"][f"v{k+1}"] = {
                    "sigma": float(S[k]),
                    "enc_r2": enc_analysis["target_r2"],
                    "enc_omega": enc_analysis["dominant_omega"],
                    "enc_fourier_conc": enc_analysis["fourier_conc"],
                    "enc_mean_f": enc_analysis["mean_f"],
                    "dec_r2": dec_analysis["target_r2"],
                    "dec_omega": dec_analysis["dominant_omega"],
                    "dec_fourier_conc": dec_analysis["fourier_conc"],
                    "dec_mean_f": dec_analysis["mean_f"],
                }

            all_results[tag][phase_name] = phase_results

    if not all_results:
        print("No checkpoints found.")
        return

    # ── Plots ──
    phases = list(phase_indices.keys())

    # Plot 1: Singular values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SCAN Gram Singular Values Across Phases", fontsize=14)
    for i, tag in enumerate(["grok", "memo"]):
        if tag not in all_results:
            continue
        ax = axes[i]
        for k in range(5):
            vals = []
            for p in phases:
                r = all_results[tag].get(p, {})
                sv = r.get("singular_values", [0]*6)
                vals.append(sv[k] if k < len(sv) else 0)
            ax.plot(range(len(phases)), vals, 'o-', label=f"σ_{k+1}", markersize=5)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=8)
        ax.set_title(f"{tag}")
        ax.set_ylabel("σ")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "gram_sv_across_phases.png", dpi=150)
    plt.close(fig)

    # Plot 2: Edge vs Bulk R² (encoder and decoder)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SCAN: Perturbation Response R² — Edge vs Bulk", fontsize=14)

    for col, (side, r2_key) in enumerate([("Encoder→Action", "enc_r2"), ("Decoder→Token", "dec_r2")]):
        for row, tag in enumerate(["grok", "memo"]):
            if tag not in all_results:
                continue
            ax = axes[row, col]
            for k, label, color in [(0, "v1 (edge)", "steelblue"), (1, "v2 (edge)", "royalblue"),
                                     (2, "v3 (bulk)", "coral"), (3, "v4 (bulk)", "lightsalmon")]:
                vals = []
                for p in phases:
                    r = all_results[tag].get(p, {})
                    d = r.get("directions", {}).get(f"v{k+1}", {})
                    vals.append(d.get(r2_key, 0))
                ax.plot(range(len(phases)), vals, 'o-', color=color, label=label, markersize=6)
            ax.set_xticks(range(len(phases)))
            ax.set_xticklabels(phases, fontsize=8)
            ax.set_ylabel("R²")
            ax.set_title(f"{tag}: {side}")
            ax.legend(fontsize=7)
            ax.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "gram_edge_vs_bulk_r2.png", dpi=150)
    plt.close(fig)

    # Plot 3: Fourier concentration
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SCAN: Fourier Concentration of Perturbation Response", fontsize=14)

    for col, (side, f_key, w_key) in enumerate([
        ("Encoder", "enc_fourier_conc", "enc_omega"),
        ("Decoder", "dec_fourier_conc", "dec_omega"),
    ]):
        for row, tag in enumerate(["grok", "memo"]):
            if tag not in all_results:
                continue
            ax = axes[row, col]
            for k, label, color in [(0, "v1", "steelblue"), (1, "v2", "royalblue"),
                                     (2, "v3", "coral"), (3, "v4", "lightsalmon")]:
                vals = []
                omegas = []
                for p in phases:
                    r = all_results[tag].get(p, {})
                    d = r.get("directions", {}).get(f"v{k+1}", {})
                    vals.append(d.get(f_key, 0))
                    omegas.append(d.get(w_key, 0))
                ax.plot(range(len(phases)), vals, 'o-', color=color, label=label, markersize=6)
                for j, (v, w) in enumerate(zip(vals, omegas)):
                    ax.annotate(f"ω={w}", (j, v), fontsize=5, ha='center', va='bottom')
            ax.set_xticks(range(len(phases)))
            ax.set_xticklabels(phases, fontsize=8)
            ax.set_ylabel("F_k")
            ax.set_title(f"{tag}: {side}")
            ax.legend(fontsize=7)
            ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "gram_fourier_concentration.png", dpi=150)
    plt.close(fig)

    # Save
    torch.save(all_results, FIG_DIR / "gram_edge_functional_results.pt")
    print(f"\nSaved figures to {FIG_DIR}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: SCAN Gram Edge vs Bulk Functional Modes")
    print("="*70)
    for tag in ["grok", "memo"]:
        if tag not in all_results:
            continue
        print(f"\n{'─'*50}")
        print(f"  {tag.upper()}")
        print(f"{'─'*50}")
        for phase in phases:
            r = all_results[tag].get(phase, {})
            if not r:
                continue
            print(f"\n  {phase} (step {r.get('step', '?')}):")
            print(f"    g₂₃ = {r.get('g23', 0):.6f}")
            for vname in ["v1", "v2", "v3", "v4"]:
                d = r.get("directions", {}).get(vname, {})
                if d:
                    label = "EDGE" if vname in ["v1", "v2"] else "BULK"
                    print(f"    {vname} ({label}): σ={d.get('sigma', 0):.4f}, "
                          f"enc_R²={d.get('enc_r2', 0):.3f}(ω={d.get('enc_omega', 0)}), "
                          f"dec_R²={d.get('dec_r2', 0):.3f}(ω={d.get('dec_omega', 0)})")


if __name__ == "__main__":
    main()
