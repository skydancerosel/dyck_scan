#!/usr/bin/env python3
"""
#7: Nonlinear probes — linear vs quadratic vs MLP for edge vs bulk.

Test whether grokked info is truly absent or just re-encoded nonlinearly.
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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from dyck.grok_sweep import DyckTransformerLM, VOCAB_SIZE, build_depth_dataset, split_dataset
from spectral.fourier_functional_dyck import load_model_at_step, extract_hidden_reps
from spectral.gram_edge_functional_modes import get_attn_param_vector, compute_gram_svd

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"
GRAM_WINDOW = 5


def probe_with_method(X_train, y_train, X_test, y_test, method="linear"):
    """Fit probe and return R²."""
    if method == "linear":
        probe = Ridge(alpha=1.0)
    elif method == "quadratic":
        probe = Pipeline([
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("ridge", Ridge(alpha=10.0)),
        ])
    elif method == "mlp":
        probe = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500,
                              early_stopping=True, validation_fraction=0.15,
                              random_state=42, alpha=0.01)
    else:
        raise ValueError(f"Unknown method: {method}")

    try:
        probe.fit(X_train, y_train)
        pred = probe.predict(X_test)
        return r2_score(y_test, pred)
    except Exception:
        return 0.0


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # Data
    X_all, Y_all = build_depth_dataset(n_seqs=5050, max_pairs=12, ctx_len=24, seed=0)
    _, _, test_x, test_y = split_dataset(X_all, Y_all, frac_train=50/5050, seed=0)
    test_x = test_x[:300]
    test_y = test_y[:300]
    mask = (test_y != -100).numpy()
    depths = test_y.numpy()

    phase_steps = {"pre_grok": 200, "at_grok": 600, "post_grok": 2000, "late": 5000}
    methods = ["linear", "quadratic", "mlp"]

    all_results = {}

    for tag, ckpt_name in [("grok", "dyck_grok_fourier.pt"), ("memo", "dyck_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        ckpt = torch.load(ckpt_path, weights_only=False)
        snapshots = ckpt["snapshots"]

        print(f"\n{'='*50}")
        print(f"Nonlinear probes: {tag}")
        print(f"{'='*50}")

        all_results[tag] = {}

        for phase_name, target_step in phase_steps.items():
            model, cfg, actual_step = load_model_at_step(ckpt_path, target_step)
            reps = extract_hidden_reps(model, test_x)

            print(f"\n  {phase_name} (step {actual_step}):")

            phase_results = {}

            # 1. Full layer representations
            for layer_name in ["layer_0", "layer_1"]:
                if layer_name not in reps:
                    continue
                R = reps[layer_name]
                N, T, D = R.shape
                flat = R.reshape(-1, D)
                d_flat = depths.reshape(-1)
                m_flat = mask.reshape(-1).astype(bool)
                X = flat[m_flat]
                y = d_flat[m_flat]

                n = len(X)
                perm = np.random.permutation(n)
                s = int(0.7 * n)

                for method in methods:
                    r2 = probe_with_method(X[perm[:s]], y[perm[:s]], X[perm[s:]], y[perm[s:]], method)
                    phase_results[f"{layer_name}_{method}"] = r2

                print(f"    {layer_name}: " + ", ".join(
                    f"{m}={phase_results[f'{layer_name}_{m}']:.3f}" for m in methods))

            # 2. Gram edge/bulk projections
            snap_idx = min(range(len(snapshots)), key=lambda i: abs(snapshots[i]["step"] - actual_step))
            gram = compute_gram_svd(snapshots, snap_idx, GRAM_WINDOW)

            if gram is not None and "layer_1" in reps:
                Vh = gram["Vh"]
                R = reps["layer_1"]
                N, T, D = R.shape

                # Project onto each Gram direction
                for k in range(min(4, Vh.shape[0])):
                    v_k = Vh[k]
                    # v_k is in attn param space, not hidden space
                    # Instead project hidden reps onto SVD of W_Q right singular vectors
                    pass  # Skip Gram projection probing (different spaces)

                # Use W_Q/W_K SVD directions instead
                layer = model.encoder.layers[1]
                attn = layer.self_attn
                d = attn.embed_dim
                if attn._qkv_same_embed_dim:
                    Wq = attn.in_proj_weight[:d].detach().cpu().numpy()
                U_q, S_q, Vh_q = np.linalg.svd(Wq, full_matrices=False)

                for k in range(min(4, Vh_q.shape[0])):
                    v_k = Vh_q[k]  # [D]
                    # Project layer_1 reps onto this direction: scalar per position
                    proj = np.einsum('ntd,d->nt', R, v_k)  # [N, T]
                    proj_flat = proj.reshape(-1)[m_flat].reshape(-1, 1)

                    for method in methods:
                        r2 = probe_with_method(
                            proj_flat[perm[:s]], y[perm[:s]],
                            proj_flat[perm[s:]], y[perm[s:]], method)
                        label = "edge" if k < 2 else "bulk"
                        phase_results[f"WQ_v{k+1}_{label}_{method}"] = r2

                    label = "edge" if k < 2 else "bulk"
                    print(f"    WQ v{k+1} ({label}): " + ", ".join(
                        f"{m}={phase_results.get(f'WQ_v{k+1}_{label}_{m}', 0):.3f}" for m in methods))

            all_results[tag][phase_name] = phase_results

    # ── Plot 1: Linear vs Quadratic vs MLP (layer_1) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Depth Probe R²: Linear vs Quadratic vs MLP (Layer 1)", fontsize=14)

    phases = list(phase_steps.keys())
    method_colors = {"linear": "steelblue", "quadratic": "orange", "mlp": "darkred"}

    for i, tag in enumerate(["grok", "memo"]):
        ax = axes[i]
        for method, color in method_colors.items():
            vals = [all_results[tag][p].get(f"layer_1_{method}", 0) for p in phases]
            ax.plot(range(len(phases)), vals, 'o-', color=color, label=method, markersize=6)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=8)
        ax.set_ylabel("R²")
        ax.set_title(f"{tag}")
        ax.legend()
        ax.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "nonlinear_probes_layer1.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Edge vs Bulk with nonlinear probes ──
    fig, axes = plt.subplots(2, len(phases), figsize=(5*len(phases), 8))
    fig.suptitle("Edge vs Bulk: Linear / Quadratic / MLP Probes", fontsize=14)

    for col, phase in enumerate(phases):
        for row, tag in enumerate(["grok", "memo"]):
            ax = axes[row, col]
            for k in range(4):
                label = "edge" if k < 2 else "bulk"
                for method, marker in [("linear", "o"), ("quadratic", "s"), ("mlp", "^")]:
                    key = f"WQ_v{k+1}_{label}_{method}"
                    val = all_results[tag][phase].get(key, 0)
                    color = "steelblue" if k < 2 else "coral"
                    alpha = {"linear": 1.0, "quadratic": 0.7, "mlp": 0.5}[method]
                    ax.bar(k * 3 + ["linear", "quadratic", "mlp"].index(method),
                           val, color=color, alpha=alpha)

            ax.set_title(f"{tag} {phase}", fontsize=9)
            ax.set_ylabel("R²")
            ax.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "nonlinear_probes_edge_bulk.png", dpi=150)
    plt.close(fig)

    # Save
    torch.save(all_results, FIG_DIR / "nonlinear_probes_results.pt")

    print(f"\nSaved figures to {FIG_DIR}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Nonlinear Probes")
    print("="*70)
    for tag in ["grok", "memo"]:
        print(f"\n  {tag}:")
        for phase in phases:
            r = all_results[tag][phase]
            print(f"    {phase}:")
            for ln in ["layer_1"]:
                print(f"      {ln}: linear={r.get(f'{ln}_linear',0):.3f}, "
                      f"quad={r.get(f'{ln}_quadratic',0):.3f}, "
                      f"mlp={r.get(f'{ln}_mlp',0):.3f}")


if __name__ == "__main__":
    main()
