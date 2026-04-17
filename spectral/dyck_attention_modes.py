#!/usr/bin/env python3
"""
Attention pattern spectral analysis for Dyck-1.

For each attention head, extract attention patterns on test sequences
and compute their DFT to identify spectral structure.

Key questions:
- Do attention heads specialize (e.g., "counting" vs "matching")?
- Does spectral structure in attention emerge at grokking?
- What frequency modes dominate each head?
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

from dyck.grok_sweep import DyckTransformerLM, VOCAB_SIZE, build_depth_dataset, split_dataset
from spectral.fourier_functional_dyck import load_model_at_step

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"


def extract_attention_patterns(model, X, device="cpu"):
    """Extract attention weight matrices for all heads on input X.

    Returns dict: (layer, head) -> [N, T, T] attention patterns
    """
    model = model.to(device)
    model.eval()

    attention_maps = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            # nn.MultiheadAttention returns (attn_output, attn_weights)
            # when need_weights=True
            pass
        return hook_fn

    # Use a different approach: manually compute attention
    with torch.no_grad():
        X_dev = X.to(device)
        B, T = X_dev.shape
        pos = torch.arange(T, device=device)
        h = model.tok_emb(X_dev) + model.pos_emb(pos)[None, :, :]

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        for layer_idx, layer in enumerate(model.encoder.layers):
            attn = layer.self_attn
            d = attn.embed_dim
            n_heads = attn.num_heads
            d_head = d // n_heads

            # Apply pre-norm (norm_first=True)
            h_normed = layer.norm1(h)

            # Extract Q, K, V projections
            if attn._qkv_same_embed_dim:
                Wq = attn.in_proj_weight[:d]
                Wk = attn.in_proj_weight[d:2*d]
                Wv = attn.in_proj_weight[2*d:]
                bq = attn.in_proj_bias[:d] if attn.in_proj_bias is not None else None
                bk = attn.in_proj_bias[d:2*d] if attn.in_proj_bias is not None else None
                bv = attn.in_proj_bias[2*d:] if attn.in_proj_bias is not None else None
            else:
                Wq, Wk, Wv = attn.q_proj_weight, attn.k_proj_weight, attn.v_proj_weight
                bq, bk, bv = None, None, None

            Q = torch.nn.functional.linear(h_normed, Wq, bq)  # [B, T, d]
            K = torch.nn.functional.linear(h_normed, Wk, bk)
            V = torch.nn.functional.linear(h_normed, Wv, bv)

            # Reshape for multi-head: [B, T, n_heads, d_head] -> [B, n_heads, T, d_head]
            Q = Q.view(B, T, n_heads, d_head).transpose(1, 2)
            K = K.view(B, T, n_heads, d_head).transpose(1, 2)
            V = V.view(B, T, n_heads, d_head).transpose(1, 2)

            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)  # [B, n_heads, T, T]

            # Apply causal mask
            scores = scores + causal_mask[None, None, :, :]

            attn_weights = torch.softmax(scores, dim=-1)  # [B, n_heads, T, T]

            for head in range(n_heads):
                attention_maps[(layer_idx, head)] = attn_weights[:, head].cpu().numpy()

            # Forward through the full layer to get output for next layer
            h = layer(h, src_mask=causal_mask, is_causal=True)

    return attention_maps


def analyze_attention_spectra(attn_maps, mask):
    """Compute spectral analysis of attention patterns.

    For each head, compute:
    - Mean attention pattern
    - DFT of attention rows
    - Entropy of attention distribution
    - "Counting" score (how uniform is backward attention?)
    """
    results = {}

    for (layer, head), attn in attn_maps.items():
        N, T, T2 = attn.shape
        assert T == T2

        # Mean attention pattern
        mean_attn = attn.mean(axis=0)  # [T, T]

        # DFT of each row of attention pattern (what frequencies in "what to attend to")
        # For each query position t, attention[t, :t+1] is a distribution
        # We compute FFT of mean attention rows
        fft_rows = np.fft.rfft(mean_attn, axis=1)  # [T, T//2+1]
        power = np.abs(fft_rows) ** 2
        mean_power = power.mean(axis=0)  # Average over query positions

        # Attention entropy per position (averaged over sequences)
        # H = -sum(p * log(p))
        eps = 1e-10
        entropy_per_pos = -(attn * np.log(attn + eps)).sum(axis=-1).mean(axis=0)  # [T]
        mean_entropy = entropy_per_pos.mean()

        # "Counting" score: how uniform is attention over valid past positions?
        # Perfect counter: attn[t, :t+1] = 1/(t+1) for all t
        counting_score = 0.0
        for t in range(T):
            if t == 0:
                continue
            uniform = np.ones(t + 1) / (t + 1)
            actual = mean_attn[t, :t+1]
            # KL divergence from uniform
            kl = np.sum(actual * np.log((actual + eps) / (uniform + eps)))
            counting_score += kl
        counting_score /= max(T - 1, 1)

        # "Local" score: how much attention is on the current position?
        local_score = np.mean([mean_attn[t, t] for t in range(T)])

        # "Recent" score: attention on last 3 positions
        recent_score = np.mean([mean_attn[t, max(0, t-2):t+1].sum() for t in range(T)])

        results[(layer, head)] = {
            "mean_attn": mean_attn,
            "fft_power": mean_power,
            "mean_entropy": mean_entropy,
            "counting_kl": counting_score,  # lower = more uniform = more counting
            "local_score": local_score,
            "recent_score": recent_score,
        }

    return results


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Generate test data
    X_all, Y_all = build_depth_dataset(n_seqs=5050, max_pairs=12, ctx_len=24, seed=0)
    _, _, test_x, test_y = split_dataset(X_all, Y_all, frac_train=50/5050, seed=0)
    test_x = test_x[:200]  # Smaller batch (attention maps are big)
    test_y = test_y[:200]
    mask = (test_y != -100).numpy()

    analysis_steps = [0, 500, 1000, 2000, 5000]

    all_results = {}

    for tag, ckpt_name in [("grok", "dyck_grok_fourier.pt"), ("memo", "dyck_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        print(f"\n{'='*50}")
        print(f"Attention modes: {tag}")
        print(f"{'='*50}")

        all_results[tag] = {}

        for target_step in analysis_steps:
            model, cfg, actual_step = load_model_at_step(ckpt_path, target_step)
            attn_maps = extract_attention_patterns(model, test_x)
            analysis = analyze_attention_spectra(attn_maps, mask)
            all_results[tag][actual_step] = analysis

            print(f"\n  Step {actual_step}:")
            for (layer, head), r in sorted(analysis.items()):
                print(f"    L{layer}H{head}: entropy={r['mean_entropy']:.2f}, "
                      f"counting_KL={r['counting_kl']:.3f}, "
                      f"local={r['local_score']:.3f}, "
                      f"recent={r['recent_score']:.3f}")

    # ── Plot 1: Mean attention patterns (grok vs memo, final step) ──
    fig, axes = plt.subplots(2, 8, figsize=(24, 7))
    fig.suptitle("Mean Attention Patterns: Grokked vs Memorized (Final Step)", fontsize=14)

    for row, tag in enumerate(["grok", "memo"]):
        steps = sorted(all_results[tag].keys())
        late = steps[-1]
        col = 0
        for layer in range(2):
            for head in range(4):
                ax = axes[row, col]
                r = all_results[tag][late][(layer, head)]
                im = ax.imshow(r["mean_attn"], aspect='auto', cmap='viridis')
                ax.set_title(f"{tag} L{layer}H{head}\nent={r['mean_entropy']:.1f}", fontsize=8)
                if col == 0:
                    ax.set_ylabel("query pos")
                ax.set_xlabel("key pos", fontsize=7)
                col += 1

    plt.tight_layout()
    fig.savefig(FIG_DIR / "attention_patterns_grok_vs_memo.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {FIG_DIR / 'attention_patterns_grok_vs_memo.png'}")

    # ── Plot 2: Attention FFT power spectra per head ──
    fig, axes = plt.subplots(2, 8, figsize=(24, 7))
    fig.suptitle("Attention Pattern FFT Power Spectra", fontsize=14)

    for row, tag in enumerate(["grok", "memo"]):
        steps = sorted(all_results[tag].keys())
        late = steps[-1]
        col = 0
        for layer in range(2):
            for head in range(4):
                ax = axes[row, col]
                r = all_results[tag][late][(layer, head)]
                omega = np.arange(len(r["fft_power"]))
                ax.bar(omega, r["fft_power"],
                       color="steelblue" if tag == "grok" else "coral", alpha=0.8)
                ax.set_title(f"{tag} L{layer}H{head}", fontsize=8)
                ax.set_xlabel("ω", fontsize=7)
                col += 1

    plt.tight_layout()
    fig.savefig(FIG_DIR / "attention_fft_power.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'attention_fft_power.png'}")

    # ── Plot 3: Head specialization scores over training (grok only) ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Attention Head Specialization During Training (Grokked)", fontsize=14)

    metrics = ["mean_entropy", "counting_kl", "local_score", "recent_score"]
    titles = ["Entropy", "Counting KL (↓=more uniform)", "Local Score", "Recent Score"]

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        steps_sorted = sorted(all_results["grok"].keys())

        for layer in range(2):
            for head in range(4):
                vals = [all_results["grok"][s][(layer, head)][metric] for s in steps_sorted]
                ax.plot(steps_sorted, vals, 'o-', label=f"L{layer}H{head}",
                        markersize=3, alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel(metric)
        if idx == 0:
            ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "attention_head_specialization.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'attention_head_specialization.png'}")

    # ── Plot 4: Entropy comparison grok vs memo ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_title("Mean Attention Entropy: Grokked vs Memorized", fontsize=14)

    for tag, ls, color in [("grok", "-", "steelblue"), ("memo", "--", "coral")]:
        steps_sorted = sorted(all_results[tag].keys())
        for layer in range(2):
            for head in range(4):
                vals = [all_results[tag][s][(layer, head)]["mean_entropy"] for s in steps_sorted]
                label = f"{tag} L{layer}H{head}" if layer == 0 and head == 0 else None
                ax.plot(steps_sorted, vals, f'o{ls}', color=color,
                        alpha=0.5, markersize=3)
        # Plot mean across heads
        mean_vals = []
        for s in steps_sorted:
            mean_vals.append(np.mean([all_results[tag][s][(l, h)]["mean_entropy"]
                                      for l in range(2) for h in range(4)]))
        ax.plot(steps_sorted, mean_vals, f's{ls}', color=color,
                label=f"{tag} (mean)", markersize=6, linewidth=2)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean attention entropy")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "attention_entropy_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'attention_entropy_comparison.png'}")

    # Save results (without large attention matrices)
    save_results = {}
    for tag in all_results:
        save_results[tag] = {}
        for step in all_results[tag]:
            save_results[tag][step] = {}
            for key, val in all_results[tag][step].items():
                save_results[tag][step][key] = {
                    k: v for k, v in val.items() if k != "mean_attn"
                }
    torch.save(save_results, FIG_DIR / "attention_modes_results.pt")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Attention Pattern Spectral Analysis")
    print("="*60)
    for tag in ["grok", "memo"]:
        steps = sorted(all_results[tag].keys())
        late = steps[-1]
        print(f"\n{tag} (step {late}):")
        for layer in range(2):
            for head in range(4):
                r = all_results[tag][late][(layer, head)]
                print(f"  L{layer}H{head}: entropy={r['mean_entropy']:.2f}, "
                      f"counting_KL={r['counting_kl']:.3f}, "
                      f"local={r['local_score']:.3f}, recent={r['recent_score']:.3f}")


if __name__ == "__main__":
    main()
