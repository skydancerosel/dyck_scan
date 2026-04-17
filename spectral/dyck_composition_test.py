#!/usr/bin/env python3
"""
Compositional structure test for Dyck-1 depth prediction.

Analogous to x2y2_composition_test.py — tests whether the model
computes depth compositionally via:
    depth(t) = cumsum of local contributions (+1 for open, -1 for close)

We decompose prediction into:
  (a) Token identity features: does the model encode open/close distinction?
  (b) Positional accumulation: does it track running depth?
  (c) Cross-terms: do token × position interactions explain depth?

Key finding expected: the grokked model factors depth into
"what token am I" × "cumulative counting" while the memorized
model uses more entangled representations.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from dyck.grok_sweep import (
    DyckTransformerLM, VOCAB_SIZE, build_depth_dataset, split_dataset,
    TOK_OPEN, TOK_CLOSE, TOK_PAD
)
from spectral.fourier_functional_dyck import load_model_at_step, extract_hidden_reps

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"


def build_composition_features(tokens, depths, mask):
    """Build compositional feature sets for probing.

    Features:
    1. token_feat: one-hot encoding of token identity [is_open, is_close]
    2. position_feat: position index (normalized)
    3. cumsum_feat: running sum of token contributions (+1/-1)
    4. cross_feat: token × cumsum interaction features

    Returns flattened valid-only arrays.
    """
    N, T = tokens.shape
    mask_flat = mask.reshape(-1).astype(bool)

    # Token features: +1 for open, -1 for close, 0 for pad
    token_sign = np.zeros((N, T), dtype=np.float32)
    token_sign[tokens == TOK_OPEN] = 1.0
    token_sign[tokens == TOK_CLOSE] = -1.0

    # One-hot token features
    is_open = (tokens == TOK_OPEN).astype(np.float32)
    is_close = (tokens == TOK_CLOSE).astype(np.float32)
    token_feat = np.stack([is_open, is_close], axis=-1)  # [N, T, 2]

    # Position features (normalized)
    pos = np.arange(T, dtype=np.float32)[None, :].repeat(N, axis=0) / T
    pos_feat = pos[:, :, None]  # [N, T, 1]

    # Cumulative sum features (the "ground truth" compositional structure)
    cumsum = np.cumsum(token_sign, axis=1)  # [N, T]
    cumsum_feat = cumsum[:, :, None]  # [N, T, 1]

    # Cross features: token × cumsum, token × position
    cross_feat = np.concatenate([
        token_feat * cumsum_feat,          # [N, T, 2]: token_type × running_depth
        token_feat * pos_feat,             # [N, T, 2]: token_type × position
        cumsum_feat * pos_feat,            # [N, T, 1]: running_depth × position
    ], axis=-1)  # [N, T, 5]

    depths_flat = depths.reshape(-1)

    return {
        "token": token_feat.reshape(-1, 2)[mask_flat],
        "position": pos_feat.reshape(-1, 1)[mask_flat],
        "cumsum": cumsum_feat.reshape(-1, 1)[mask_flat],
        "cross": cross_feat.reshape(-1, 5)[mask_flat],
        "depth": depths_flat[mask_flat],
    }


def composition_probe(reps, tokens, depths, mask):
    """Test compositional structure in representations.

    Returns R² for different feature combinations predicting depth
    from the learned representations.
    """
    N, T, D = reps.shape
    mask_flat = mask.reshape(-1).astype(bool)
    reps_flat = reps.reshape(-1, D)[mask_flat]
    depths_flat = depths.reshape(-1)[mask_flat]

    comp = build_composition_features(tokens, depths, mask)

    # Train/test split
    n = len(reps_flat)
    perm = np.random.permutation(n)
    split = int(0.7 * n)
    tr, te = perm[:split], perm[split:]

    results = {}

    # 1. Full representation → depth
    probe = Ridge(alpha=1.0)
    probe.fit(reps_flat[tr], depths_flat[tr])
    results["full_rep→depth"] = r2_score(depths_flat[te], probe.predict(reps_flat[te]))

    # 2. Token features only → depth
    probe.fit(comp["token"][tr], comp["depth"][tr])
    results["token→depth"] = r2_score(comp["depth"][te], probe.predict(comp["token"][te]))

    # 3. Position only → depth
    probe.fit(comp["position"][tr], comp["depth"][tr])
    results["position→depth"] = r2_score(comp["depth"][te], probe.predict(comp["position"][te]))

    # 4. Cumsum only → depth (should be ~1.0, it IS depth)
    probe.fit(comp["cumsum"][tr], comp["depth"][tr])
    results["cumsum→depth"] = r2_score(comp["depth"][te], probe.predict(comp["cumsum"][te]))

    # 5. Token + position (no cumsum) → depth
    feat_tp = np.concatenate([comp["token"], comp["position"]], axis=-1)
    probe.fit(feat_tp[tr], comp["depth"][tr])
    results["token+pos→depth"] = r2_score(comp["depth"][te], probe.predict(feat_tp[te]))

    # 6. Token + position + cross → depth
    feat_all = np.concatenate([comp["token"], comp["position"], comp["cross"]], axis=-1)
    probe.fit(feat_all[tr], comp["depth"][tr])
    results["token+pos+cross→depth"] = r2_score(comp["depth"][te], probe.predict(feat_all[te]))

    # 7. Representation → cumsum (can the model's features predict running sum?)
    probe.fit(reps_flat[tr], comp["cumsum"][tr].ravel())
    results["rep→cumsum"] = r2_score(comp["cumsum"][te].ravel(), probe.predict(reps_flat[te]))

    # 8. Representation → token_sign (can features predict local ±1?)
    token_sign = comp["token"][:, 0] - comp["token"][:, 1]  # +1 open, -1 close
    probe.fit(reps_flat[tr], token_sign[tr])
    results["rep→token_sign"] = r2_score(token_sign[te], probe.predict(reps_flat[te]))

    return results


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # Generate test data
    X_all, Y_all = build_depth_dataset(n_seqs=5050, max_pairs=12, ctx_len=24, seed=0)
    _, _, test_x, test_y = split_dataset(X_all, Y_all, frac_train=50/5050, seed=0)
    test_x = test_x[:500]
    test_y = test_y[:500]
    mask = (test_y != -100).numpy()
    depths = test_y.numpy()
    tokens = test_x.numpy()

    analysis_steps = [0, 200, 500, 1000, 2000, 5000]

    all_results = {}

    for tag, ckpt_name in [("grok", "dyck_grok_fourier.pt"), ("memo", "dyck_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        print(f"\n{'='*50}")
        print(f"Composition test: {tag}")
        print(f"{'='*50}")

        all_results[tag] = {}

        for target_step in analysis_steps:
            model, cfg, actual_step = load_model_at_step(ckpt_path, target_step)
            reps = extract_hidden_reps(model, test_x)

            step_results = {}
            for layer_name in ["layer_0", "layer_1"]:
                if layer_name not in reps:
                    continue
                comp_result = composition_probe(reps[layer_name], tokens, depths, mask)
                step_results[layer_name] = comp_result

            all_results[tag][actual_step] = step_results

            if actual_step in [0, 500, 1000, 5000]:
                print(f"\n  Step {actual_step}, layer_1:")
                for k, v in step_results.get("layer_1", {}).items():
                    print(f"    {k}: R²={v:.3f}")

    # ── Plot 1: Composition R² comparison (bar chart, final step) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    feature_keys = [
        "token→depth", "position→depth", "cumsum→depth",
        "token+pos→depth", "token+pos+cross→depth",
        "full_rep→depth", "rep→cumsum", "rep→token_sign"
    ]
    short_labels = [
        "token", "position", "cumsum",
        "tok+pos", "tok+pos+cross",
        "full_rep", "rep→cumsum", "rep→tok_sign"
    ]

    for i, tag in enumerate(["grok", "memo"]):
        ax = axes[i]
        steps = sorted(all_results[tag].keys())
        late = steps[-1]
        r = all_results[tag][late].get("layer_1", {})
        vals = [r.get(k, 0) for k in feature_keys]
        colors = ["lightcoral", "lightblue", "gold",
                   "lightsalmon", "orange",
                   "steelblue", "mediumseagreen", "plum"]
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(short_labels, fontsize=9)
        ax.set_xlabel("R²")
        ax.set_title(f"{tag} step={late} (layer_1)")
        ax.set_xlim(-0.1, 1.1)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "composition_r2_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {FIG_DIR / 'composition_r2_comparison.png'}")

    # ── Plot 2: Cross-term R² boost over training ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_title("Compositional Cross-Term R² Boost During Training", fontsize=14)

    for tag, color, ls in [("grok", "steelblue", "-"), ("memo", "coral", "--")]:
        steps_sorted = sorted(all_results[tag].keys())
        boost_vals = []
        for s in steps_sorted:
            r = all_results[tag][s].get("layer_1", {})
            base = r.get("token+pos→depth", 0)
            cross = r.get("token+pos+cross→depth", 0)
            boost_vals.append(cross - base)
        ax.plot(steps_sorted, boost_vals, f'o{ls}', color=color,
                label=f"{tag}: cross boost", markersize=4)

    ax.set_xlabel("Training step")
    ax.set_ylabel("R² boost from cross-terms")
    ax.legend()
    ax.axhline(y=0, ls='--', color='gray', alpha=0.5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "composition_cross_term_boost.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'composition_cross_term_boost.png'}")

    # ── Plot 3: rep→cumsum and rep→token_sign over training ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for j, key in enumerate(["rep→cumsum", "rep→token_sign"]):
        ax = axes[j]
        for tag, color, ls in [("grok", "steelblue", "-"), ("memo", "coral", "--")]:
            steps_sorted = sorted(all_results[tag].keys())
            vals = [all_results[tag][s].get("layer_1", {}).get(key, 0) for s in steps_sorted]
            ax.plot(steps_sorted, vals, f'o{ls}', color=color, label=tag, markersize=4)
        ax.set_title(key)
        ax.set_xlabel("Training step")
        ax.set_ylabel("R²")
        ax.legend()
        ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "composition_factorization.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'composition_factorization.png'}")

    # Save results
    torch.save(all_results, FIG_DIR / "composition_results.pt")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Compositional Structure Test")
    print("="*60)
    for tag in ["grok", "memo"]:
        steps = sorted(all_results[tag].keys())
        late = steps[-1]
        r = all_results[tag][late].get("layer_1", {})
        print(f"\n{tag} (step {late}, layer_1):")
        for k in feature_keys:
            print(f"  {k}: R²={r.get(k, 0):.3f}")


if __name__ == "__main__":
    main()
