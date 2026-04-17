#!/usr/bin/env python3
"""
Attention pattern spectral analysis for SCAN encoder-decoder.

Analyzes encoder self-attention, decoder self-attention, and cross-attention
patterns. For each head: DFT, entropy, and specialization scores.
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

from spectral.fourier_functional_scan import load_scan_model_at_step

CKPT_DIR = Path(__file__).resolve().parent / "fourier_scan_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_scan_plots"


def extract_scan_attention_patterns(model, src, tgt_in, device="cpu"):
    """Extract attention patterns from encoder/decoder/cross-attention.

    Returns dict: (type, layer, head) -> [N, T_q, T_k] patterns
    """
    model = model.to(device)
    model.eval()
    attention_maps = {}

    with torch.no_grad():
        src_dev = src.to(device)
        tgt_dev = tgt_in.to(device)
        B, S = src_dev.shape
        _, T = tgt_dev.shape

        pad_id = 0
        src_pad_mask = (src_dev == pad_id)
        tgt_pad_mask = (tgt_dev == pad_id)

        # Compute encoder embeddings
        src_pos = torch.arange(S, device=device)
        src_emb = model.src_tok_emb(src_dev) + model.src_pos_emb(src_pos)[None, :, :]

        tgt_pos = torch.arange(T, device=device)
        tgt_emb = model.tgt_tok_emb(tgt_dev) + model.tgt_pos_emb(tgt_pos)[None, :, :]

        d_model = model.d_model
        n_heads = model.transformer.nhead
        d_head = d_model // n_heads

        # ── Encoder self-attention ──
        h_enc = src_emb
        for layer_idx, layer in enumerate(model.transformer.encoder.layers):
            attn = layer.self_attn

            h_normed = layer.norm1(h_enc)
            if attn._qkv_same_embed_dim:
                Wq = attn.in_proj_weight[:d_model]
                Wk = attn.in_proj_weight[d_model:2*d_model]
                bq = attn.in_proj_bias[:d_model] if attn.in_proj_bias is not None else None
                bk = attn.in_proj_bias[d_model:2*d_model] if attn.in_proj_bias is not None else None

            Q = nn.functional.linear(h_normed, Wq, bq).view(B, S, n_heads, d_head).transpose(1, 2)
            K = nn.functional.linear(h_normed, Wk, bk).view(B, S, n_heads, d_head).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)
            # Apply src padding mask
            if src_pad_mask is not None:
                scores = scores.masked_fill(src_pad_mask[:, None, None, :], float('-inf'))

            attn_weights = torch.softmax(scores, dim=-1)
            for head in range(n_heads):
                attention_maps[("enc_self", layer_idx, head)] = attn_weights[:, head].cpu().numpy()

            # Forward through full layer
            h_enc = layer(h_enc, src_key_padding_mask=src_pad_mask)

        # ── Decoder self-attention + cross-attention ──
        memory = h_enc  # encoder output
        h_dec = tgt_emb
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        for layer_idx, layer in enumerate(model.transformer.decoder.layers):
            # Self-attention
            attn = layer.self_attn
            h_normed = layer.norm1(h_dec)
            if attn._qkv_same_embed_dim:
                Wq = attn.in_proj_weight[:d_model]
                Wk = attn.in_proj_weight[d_model:2*d_model]
                bq = attn.in_proj_bias[:d_model] if attn.in_proj_bias is not None else None
                bk = attn.in_proj_bias[d_model:2*d_model] if attn.in_proj_bias is not None else None

            Q = nn.functional.linear(h_normed, Wq, bq).view(B, T, n_heads, d_head).transpose(1, 2)
            K = nn.functional.linear(h_normed, Wk, bk).view(B, T, n_heads, d_head).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)
            scores = scores + tgt_mask[None, None, :, :]
            if tgt_pad_mask is not None:
                scores = scores.masked_fill(tgt_pad_mask[:, None, None, :], float('-inf'))

            attn_weights = torch.softmax(scores, dim=-1)
            for head in range(n_heads):
                attention_maps[("dec_self", layer_idx, head)] = attn_weights[:, head].cpu().numpy()

            # Cross-attention
            xattn = layer.multihead_attn
            h_normed2 = layer.norm2(h_dec)  # norm before cross-attention
            if xattn._qkv_same_embed_dim:
                XWq = xattn.in_proj_weight[:d_model]
                XWk = xattn.in_proj_weight[d_model:2*d_model]
                xbq = xattn.in_proj_bias[:d_model] if xattn.in_proj_bias is not None else None
                xbk = xattn.in_proj_bias[d_model:2*d_model] if xattn.in_proj_bias is not None else None

            XQ = nn.functional.linear(h_normed2, XWq, xbq).view(B, T, n_heads, d_head).transpose(1, 2)
            XK = nn.functional.linear(memory, XWk, xbk).view(B, S, n_heads, d_head).transpose(1, 2)

            xscores = torch.matmul(XQ, XK.transpose(-2, -1)) / (d_head ** 0.5)
            if src_pad_mask is not None:
                xscores = xscores.masked_fill(src_pad_mask[:, None, None, :], float('-inf'))

            xattn_weights = torch.softmax(xscores, dim=-1)
            for head in range(n_heads):
                attention_maps[("cross", layer_idx, head)] = xattn_weights[:, head].cpu().numpy()

            # Forward through full layer
            h_dec = layer(h_dec, memory, tgt_mask=tgt_mask,
                         tgt_key_padding_mask=tgt_pad_mask,
                         memory_key_padding_mask=src_pad_mask)

    return attention_maps


def analyze_attention(attn_maps):
    """Compute entropy and specialization scores for each attention head."""
    results = {}
    eps = 1e-10

    for key, attn in attn_maps.items():
        N, T_q, T_k = attn.shape
        mean_attn = np.nanmean(attn, axis=0)  # [T_q, T_k]

        # Replace NaN with uniform (from -inf masking on all-pad)
        nan_rows = np.isnan(mean_attn).any(axis=-1)
        if nan_rows.any():
            mean_attn[nan_rows] = 1.0 / T_k

        # Entropy
        safe_attn = np.clip(attn, eps, 1.0)
        entropy_per_pos = -(safe_attn * np.log(safe_attn)).sum(axis=-1).mean(axis=0)
        mean_entropy = np.nanmean(entropy_per_pos)

        # DFT power of mean attention
        fft_rows = np.fft.rfft(mean_attn, axis=1)
        power = np.abs(fft_rows) ** 2
        mean_power = np.nanmean(power, axis=0)

        # Locality score
        local_score = np.nanmean([mean_attn[t, t] if t < T_k else 0 for t in range(T_q)])

        results[key] = {
            "mean_entropy": float(mean_entropy),
            "fft_power": mean_power,
            "local_score": float(local_score),
            "dominant_omega": int(np.argmax(mean_power)),
        }

    return results


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    analysis_steps = [0, 2000, 5000, 10000]

    all_results = {}

    for tag, ckpt_name in [("grok", "scan_grok_fourier.pt"), ("memo", "scan_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"Skipping {tag}")
            continue

        ckpt = torch.load(ckpt_path, weights_only=False)
        test_src = ckpt["test_src"][:100]
        test_tgt_in = ckpt["test_tgt_in"][:100]

        print(f"\n{'='*50}")
        print(f"Attention modes: {tag}")
        print(f"{'='*50}")

        all_results[tag] = {}

        for target_step in analysis_steps:
            model, _, actual_step = load_scan_model_at_step(ckpt_path, target_step)
            attn_maps = extract_scan_attention_patterns(model, test_src, test_tgt_in)
            analysis = analyze_attention(attn_maps)
            all_results[tag][actual_step] = analysis

            print(f"\n  Step {actual_step}:")
            for key in sorted(analysis.keys()):
                r = analysis[key]
                attn_type, layer, head = key
                print(f"    {attn_type} L{layer}H{head}: entropy={r['mean_entropy']:.2f}, "
                      f"local={r['local_score']:.3f}, ω*={r['dominant_omega']}")

    if not all_results:
        print("No checkpoints found.")
        return

    # ── Plot 1: Entropy comparison across attention types ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Attention Entropy: Grokked vs Memorized", fontsize=14)

    for col, attn_type in enumerate(["enc_self", "dec_self", "cross"]):
        ax = axes[col]
        for tag, color, ls in [("grok", "steelblue", "-"), ("memo", "coral", "--")]:
            if tag not in all_results:
                continue
            steps_sorted = sorted(all_results[tag].keys())
            # Average entropy across all heads of this type
            mean_ents = []
            for s in steps_sorted:
                ents = [all_results[tag][s][k]["mean_entropy"]
                        for k in all_results[tag][s] if k[0] == attn_type]
                mean_ents.append(np.mean(ents) if ents else np.nan)
            ax.plot(steps_sorted, mean_ents, f'o{ls}', color=color, label=tag, markersize=5)
        ax.set_title(f"{attn_type} (mean over heads)")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean entropy")
        ax.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "attention_entropy_by_type.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Per-head entropy for cross-attention (most interesting) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, tag in enumerate(["grok", "memo"]):
        if tag not in all_results:
            continue
        ax = axes[i]
        steps_sorted = sorted(all_results[tag].keys())
        for layer in range(3):
            for head in range(4):
                key = ("cross", layer, head)
                ents = [all_results[tag][s].get(key, {}).get("mean_entropy", np.nan)
                        for s in steps_sorted]
                ax.plot(steps_sorted, ents, 'o-', label=f"L{layer}H{head}",
                        markersize=3, alpha=0.7)
        ax.set_title(f"{tag}: Cross-Attention Entropy Per Head")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Entropy")
        if i == 0:
            ax.legend(fontsize=6, ncol=3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "cross_attention_entropy_per_head.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: FFT power for final step cross-attention ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Cross-Attention FFT Power (Layer 0, Final Step)", fontsize=14)

    for row, tag in enumerate(["grok", "memo"]):
        if tag not in all_results:
            continue
        steps = sorted(all_results[tag].keys())
        late = steps[-1]
        # Show 3 decoder layers, head 0
        for col in range(3):
            ax = axes[row, col]
            key = ("cross", col, 0)
            if key in all_results[tag][late]:
                power = all_results[tag][late][key]["fft_power"]
                omega = np.arange(len(power))
                ax.bar(omega, power, color="steelblue" if tag == "grok" else "coral", alpha=0.8)
                ent = all_results[tag][late][key]["mean_entropy"]
                ax.set_title(f"{tag} L{col}H0 (ent={ent:.2f})", fontsize=10)
            ax.set_xlabel("ω")
            ax.set_ylabel("|A(ω)|²")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "cross_attention_fft_power.png", dpi=150)
    plt.close(fig)

    # Save results (without large arrays)
    save_results = {}
    for tag in all_results:
        save_results[tag] = {}
        for step in all_results[tag]:
            save_results[tag][step] = {}
            for key, val in all_results[tag][step].items():
                save_results[tag][step][key] = {
                    k: v for k, v in val.items() if k != "fft_power"
                }
    torch.save(save_results, FIG_DIR / "attention_modes_results.pt")

    print(f"\nSaved figures to {FIG_DIR}")

    print("\n" + "="*60)
    print("SUMMARY: SCAN Attention Pattern Spectral Analysis")
    print("="*60)
    for tag in all_results:
        steps = sorted(all_results[tag].keys())
        late = steps[-1]
        print(f"\n{tag} (step {late}):")
        for attn_type in ["enc_self", "dec_self", "cross"]:
            keys = [k for k in all_results[tag][late] if k[0] == attn_type]
            ents = [all_results[tag][late][k]["mean_entropy"] for k in keys]
            locals_ = [all_results[tag][late][k]["local_score"] for k in keys]
            print(f"  {attn_type}: mean_entropy={np.mean(ents):.2f}, mean_local={np.mean(locals_):.3f}")


if __name__ == "__main__":
    main()
