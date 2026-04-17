#!/usr/bin/env python3
"""
#3: Multi-seed replication of key findings.

Retrain Dyck with seeds 137 and 2024, then verify:
  - Grad-WD decomposition (v1 flips at grokking)
  - Ablation paradox (edge critical, perturbation flat)
  - Nonlinear probe (MLP recovers R² where linear fails)
  - Depth-basis Fourier concentration
"""

import sys, os, time, random, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from dyck.grok_sweep import (
    DyckSweepConfig, DyckTransformerLM, VOCAB_SIZE,
    build_depth_dataset, split_dataset, sample_batch,
    masked_ce_loss, masked_accuracy, eval_on_dataset,
)
from spectral.gram_edge_functional_modes import (
    get_attn_param_vector, get_attn_param_keys, compute_gram_svd,
)
from spectral.fourier_correct_basis import (
    depth_grouped_fourier, compute_perturbation_response, build_model,
)

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"
GRAM_WINDOW = 5
EPS_SCALE = 0.005


def retrain_seed(seed, steps=10000, log_every=100):
    """Quick retrain for a single seed."""
    cfg = DyckSweepConfig()
    cfg.SEED = seed
    cfg.WEIGHT_DECAY = 1.0
    cfg.STEPS = steps

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    X_all, Y_all = build_depth_dataset(
        n_seqs=cfg.N_TOTAL, max_pairs=cfg.MAX_PAIRS, ctx_len=cfg.CTX_LEN, seed=cfg.DATA_SEED)
    frac = cfg.N_TRAIN / cfg.N_TOTAL
    train_x, train_y, test_x, test_y = split_dataset(X_all, Y_all, frac_train=frac, seed=cfg.DATA_SEED)

    model = DyckTransformerLM(
        vocab_size=VOCAB_SIZE, ctx_len=max(cfg.CTX_LEN, cfg.CTX_LEN_OOD),
        d_model=cfg.D_MODEL, n_layers=cfg.N_LAYERS, n_heads=cfg.N_HEADS,
        d_ff=cfg.D_FF, dropout=cfg.DROPOUT, n_classes=cfg.N_CLASSES,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
                             betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2))

    snapshots = [{"step": 0, "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()}}]
    metrics = []

    for step in range(1, steps + 1):
        model.train()
        bx, by = sample_batch(train_x, train_y, cfg.BATCH_SIZE, "cpu")
        logits = model(bx)
        loss = masked_ce_loss(logits, by)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        if step % log_every == 0:
            train_loss, train_acc = eval_on_dataset(model, train_x, train_y, "cpu")
            test_loss, test_acc = eval_on_dataset(model, test_x, test_y, "cpu")
            metrics.append({"step": step, "train_acc": train_acc, "test_acc": test_acc})
            snapshots.append({"step": step,
                              "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()}})

    return {"cfg": asdict(cfg), "snapshots": snapshots, "metrics": metrics,
            "train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}


def analyze_seed(ckpt, seed_label):
    """Run all key analyses on one seed's checkpoint."""
    snapshots = ckpt["snapshots"]
    cfg = ckpt["cfg"]
    test_x = ckpt["test_x"][:300]
    test_y = ckpt["test_y"][:300]
    mask = (test_y != -100).numpy()
    depths = test_y.numpy()

    param_keys = get_attn_param_keys(snapshots[0]["state_dict"])
    model = build_model(cfg)

    phase_indices = {"pre_grok": 2, "at_grok": 5, "post_grok": 14, "late": 39}
    phases = list(phase_indices.keys())

    results = {}

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

        # ── Grad-WD decomposition ──
        if center_idx > 0:
            theta_prev = get_attn_param_vector(snapshots[center_idx-1]["state_dict"]).numpy()
            theta_curr = get_attn_param_vector(state_dict).numpy()
            delta_total = theta_curr - theta_prev
            N_steps = snapshots[center_idx]["step"] - snapshots[center_idx-1]["step"]
            delta_wd = -cfg["LR"] * cfg["WEIGHT_DECAY"] * N_steps * (theta_prev + theta_curr) / 2.0
            delta_grad = delta_total - delta_wd

            v1 = Vh[0]
            grad_proj = np.dot(delta_grad, v1) ** 2
            wd_proj = np.dot(delta_wd, v1) ** 2
            denom = grad_proj + wd_proj
            grad_frac = grad_proj / denom if denom > 0 else 0.5
        else:
            grad_frac = 0.5

        # ── Ablation ──
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            base_logits = model(test_x)
            base_acc = masked_accuracy(base_logits, test_y)

        # Remove edge (v1+v2)
        abl_sd = {k: v.clone() for k, v in state_dict.items()}
        theta = get_attn_param_vector(state_dict).numpy()
        removal = np.zeros_like(theta)
        for ki in range(min(2, len(S))):
            v_ki = Vh[ki]
            removal += np.dot(theta, v_ki) * v_ki
        offset = 0
        for key, numel in param_keys:
            chunk = removal[offset:offset + numel]
            abl_sd[key] = abl_sd[key] - torch.tensor(chunk, dtype=abl_sd[key].dtype).reshape(abl_sd[key].shape)
            offset += numel
        model.load_state_dict(abl_sd)
        model.eval()
        with torch.no_grad():
            abl_acc = masked_accuracy(model(test_x), test_y)
        ablation_delta = abl_acc - base_acc

        # ── Perturbation KL ──
        model.load_state_dict(state_dict)
        f_k, _ = compute_perturbation_response(model, state_dict, cfg, Vh[0], param_keys, test_x, eps)
        mean_pert = f_k[mask.astype(bool)].mean()

        # ── Nonlinear probes ──
        model.load_state_dict(state_dict)
        model.eval()
        hooks_out = {}
        def hook_fn(module, inp, out, name="layer_1"):
            hooks_out[name] = out.detach().cpu().numpy()
        hook = model.encoder.layers[1].register_forward_hook(lambda m, i, o: hook_fn(m, i, o))
        with torch.no_grad():
            model(test_x)
        hook.remove()

        R = hooks_out.get("layer_1")
        if R is not None:
            N, T, D = R.shape
            flat = R.reshape(-1, D)
            d_flat = depths.reshape(-1)
            m_flat = mask.reshape(-1).astype(bool)
            X_probe = flat[m_flat]
            y_probe = d_flat[m_flat]
            n = len(X_probe)
            perm = np.random.permutation(n)
            s = int(0.7 * n)

            probe_lin = Ridge(alpha=1.0)
            probe_lin.fit(X_probe[perm[:s]], y_probe[perm[:s]])
            r2_linear = r2_score(y_probe[perm[s:]], probe_lin.predict(X_probe[perm[s:]]))

            probe_mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500,
                                      early_stopping=True, random_state=42, alpha=0.01)
            probe_mlp.fit(X_probe[perm[:s]], y_probe[perm[:s]])
            r2_mlp = r2_score(y_probe[perm[s:]], probe_mlp.predict(X_probe[perm[s:]]))
        else:
            r2_linear = r2_mlp = 0

        # ── Depth Fourier ──
        model.load_state_dict(state_dict)
        f_k_full, _ = compute_perturbation_response(model, state_dict, cfg, Vh[0], param_keys, test_x, eps)
        depth_fourier = depth_grouped_fourier(f_k_full, depths, mask)

        results[phase_name] = {
            "step": step,
            "grad_frac_v1": float(grad_frac),
            "ablation_delta_acc": float(ablation_delta),
            "base_acc": float(base_acc),
            "mean_perturbation": float(mean_pert),
            "r2_linear": float(r2_linear),
            "r2_mlp": float(r2_mlp),
            "depth_fourier_elevation": float(depth_fourier.get("elevation", 0)),
            "sigma_1": float(S[0]) if len(S) > 0 else 0,
        }

        print(f"  {phase_name} (step {step}): grad_frac={grad_frac:.3f}, "
              f"abl={ablation_delta:+.3f}, lin={r2_linear:.3f}, mlp={r2_mlp:.3f}, "
              f"F_depth={depth_fourier.get('elevation', 0):.1f}x")

    return results


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    all_seed_results = {}

    # Seed 42 already exists
    ckpt_42 = torch.load(CKPT_DIR / "dyck_grok_fourier.pt", weights_only=False)
    print("Analyzing seed 42 (existing)...")
    all_seed_results[42] = analyze_seed(ckpt_42, "s42")

    # Retrain and analyze seeds 137, 2024
    for seed in [137, 2024]:
        print(f"\n{'='*60}")
        print(f"Retraining + analyzing seed {seed}")
        print(f"{'='*60}")

        ckpt_path = CKPT_DIR / f"dyck_grok_s{seed}_fourier.pt"
        if ckpt_path.exists():
            print(f"  Loading existing {ckpt_path}")
            ckpt = torch.load(ckpt_path, weights_only=False)
        else:
            ckpt = retrain_seed(seed)
            torch.save(ckpt, ckpt_path)
            print(f"  Saved: {ckpt_path}")

        all_seed_results[seed] = analyze_seed(ckpt, f"s{seed}")

    # ── Plot: Multi-seed comparison ──
    phases = ["pre_grok", "at_grok", "post_grok", "late"]
    metrics = [
        ("grad_frac_v1", "Grad fraction on v1"),
        ("ablation_delta_acc", "Ablation Δacc (remove edge)"),
        ("r2_linear", "Linear probe R²"),
        ("r2_mlp", "MLP probe R²"),
        ("depth_fourier_elevation", "Depth Fourier elevation"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Multi-Seed Replication (Seeds 42, 137, 2024)", fontsize=14)

    colors = {42: "steelblue", 137: "coral", 2024: "forestgreen"}
    for idx, (key, ylabel) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        for seed, color in colors.items():
            vals = [all_seed_results[seed].get(p, {}).get(key, 0) for p in phases]
            ax.plot(range(len(phases)), vals, 'o-', color=color, label=f"seed {seed}",
                    markersize=6, linewidth=2)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.set_title(ylabel, fontsize=10)

    # Last panel: summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    summary = []
    for key, label in metrics:
        late_vals = [all_seed_results[s].get("late", {}).get(key, 0) for s in [42, 137, 2024]]
        grok_vals = [all_seed_results[s].get("at_grok", {}).get(key, 0) for s in [42, 137, 2024]]
        summary.append(f"{label}:")
        summary.append(f"  at_grok: {np.mean(grok_vals):.3f} ± {np.std(grok_vals):.3f}")
        summary.append(f"  late:    {np.mean(late_vals):.3f} ± {np.std(late_vals):.3f}")
    ax.text(0.05, 0.95, "\n".join(summary), transform=ax.transAxes,
            fontsize=7, family='monospace', va='top')
    ax.set_title("Summary (mean ± std)", fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "multiseed_replication.png", dpi=150)
    plt.close(fig)

    # Save
    torch.save(all_seed_results, FIG_DIR / "multiseed_results.pt")

    # Print summary
    print("\n" + "="*70)
    print("MULTI-SEED REPLICATION SUMMARY")
    print("="*70)
    for key, label in metrics:
        at_grok = [all_seed_results[s].get("at_grok", {}).get(key, 0) for s in [42, 137, 2024]]
        late = [all_seed_results[s].get("late", {}).get(key, 0) for s in [42, 137, 2024]]
        print(f"\n  {label}:")
        print(f"    at_grok: {np.mean(at_grok):.3f} ± {np.std(at_grok):.3f}  ({at_grok})")
        print(f"    late:    {np.mean(late):.3f} ± {np.std(late):.3f}  ({late})")


if __name__ == "__main__":
    main()
