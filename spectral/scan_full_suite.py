#!/usr/bin/env python3
"""
#4: SCAN versions of experiments 3-8.

Combined script: Hessian curvature, ablation, nonlinear probes, WD intervention,
random direction controls, loss decomposition — all for SCAN.
"""

import sys, os, time, random, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from scan.grok_sweep import (
    ScanTransformer, ScanSweepConfig, masked_ce_loss, masked_accuracy,
)
from spectral.gram_edge_functional_modes import get_attn_param_vector, get_attn_param_keys

CKPT_DIR = Path(__file__).resolve().parent / "fourier_scan_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_scan_plots"
GRAM_WINDOW = 5


def build_scan_model(cfg, ckpt):
    return ScanTransformer(
        src_vocab_size=ckpt["cmd_vocab"].size,
        tgt_vocab_size=ckpt["act_vocab"].size,
        max_src_len=ckpt["max_cmd_len"],
        max_tgt_len=ckpt["max_act_len"],
        d_model=cfg["D_MODEL"], n_layers=cfg["N_LAYERS"],
        n_heads=cfg["N_HEADS"], d_ff=cfg["D_FF"],
        dropout=cfg["DROPOUT"],
    )


def compute_gram_svd(snapshots, center_idx, window):
    start = max(1, center_idx - window + 1)
    end = min(len(snapshots) - 1, center_idx)
    if end - start + 1 < 2:
        return None
    deltas = []
    for i in range(start, end + 1):
        theta_prev = get_attn_param_vector(snapshots[i-1]["state_dict"]).numpy()
        theta_curr = get_attn_param_vector(snapshots[i]["state_dict"]).numpy()
        deltas.append(theta_curr - theta_prev)
    X = np.stack(deltas)
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    return {"singular_values": S, "Vh": Vh}


def ablate_directions(state_dict, param_keys, Vh, indices):
    theta = get_attn_param_vector(state_dict).numpy()
    removal = np.zeros_like(theta)
    for k in indices:
        removal += np.dot(theta, Vh[k]) * Vh[k]
    ablated = {k: v.clone() for k, v in state_dict.items()}
    offset = 0
    for key, numel in param_keys:
        chunk = removal[offset:offset + numel]
        ablated[key] = ablated[key] - torch.tensor(chunk, dtype=ablated[key].dtype).reshape(ablated[key].shape)
        offset += numel
    return ablated


def hessian_curvature(model_fn, cfg, ckpt, state_dict, param_keys, v_k,
                       src, tgt_in, tgt_out, eps=0.001):
    """v^T H v via finite difference."""
    pad_id = 0

    def eval_loss(sd):
        m = model_fn(cfg, ckpt)
        m.load_state_dict(sd)
        m.eval()
        with torch.no_grad():
            logits = m(src, tgt_in, src_pad_mask=(src == pad_id), tgt_pad_mask=(tgt_in == pad_id))
            return masked_ce_loss(logits, tgt_out).item()

    theta_norm = get_attn_param_vector(state_dict).norm().item()
    eps_s = eps * theta_norm

    L0 = eval_loss(state_dict)

    # +ε
    sd_p = {k: v.clone() for k, v in state_dict.items()}
    offset = 0
    for key, numel in param_keys:
        chunk = v_k[offset:offset + numel]
        sd_p[key] = sd_p[key] + eps_s * torch.tensor(chunk, dtype=sd_p[key].dtype).reshape(sd_p[key].shape)
        offset += numel
    Lp = eval_loss(sd_p)

    # -ε
    sd_m = {k: v.clone() for k, v in state_dict.items()}
    offset = 0
    for key, numel in param_keys:
        chunk = v_k[offset:offset + numel]
        sd_m[key] = sd_m[key] - eps_s * torch.tensor(chunk, dtype=sd_m[key].dtype).reshape(sd_m[key].shape)
        offset += numel
    Lm = eval_loss(sd_m)

    return (Lp + Lm - 2 * L0) / (eps_s ** 2), L0


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    phase_indices = {"early": 1, "pre_grok": 4, "at_grok": 9, "post_grok": 19}
    phases = list(phase_indices.keys())

    all_results = {}

    for tag, ckpt_name in [("grok", "scan_grok_fourier.pt"), ("memo", "scan_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"Skipping {tag}")
            continue

        ckpt = torch.load(ckpt_path, weights_only=False)
        snapshots = ckpt["snapshots"]
        cfg = ckpt["cfg"]
        param_keys = get_attn_param_keys(snapshots[0]["state_dict"])

        test_src = ckpt["test_src"][:150]
        test_tgt_in = ckpt["test_tgt_in"][:150]
        test_tgt_out = ckpt["test_tgt_out"][:150]
        tgt_mask = (test_tgt_out != -100).numpy()
        tgt_tokens = test_tgt_out.numpy()
        pad_id = 0

        print(f"\n{'='*60}")
        print(f"SCAN full suite: {tag}")
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

            print(f"\n  {phase_name} (step {step}):")

            # ── Base eval ──
            model = build_scan_model(cfg, ckpt)
            model.load_state_dict(state_dict)
            model.eval()
            with torch.no_grad():
                logits = model(test_src, test_tgt_in,
                              src_pad_mask=(test_src == pad_id),
                              tgt_pad_mask=(test_tgt_in == pad_id))
                base_loss = masked_ce_loss(logits, test_tgt_out).item()
                base_acc = masked_accuracy(logits, test_tgt_out)
            print(f"    Base: loss={base_loss:.4f}, acc={base_acc:.3f}")

            # ── Ablation ──
            sd_no_edge = ablate_directions(state_dict, param_keys, Vh, [0, 1])
            model.load_state_dict(sd_no_edge)
            model.eval()
            with torch.no_grad():
                ne_acc = masked_accuracy(model(test_src, test_tgt_in,
                    src_pad_mask=(test_src==pad_id), tgt_pad_mask=(test_tgt_in==pad_id)), test_tgt_out)
            edge_delta = ne_acc - base_acc

            n_dirs = min(4, len(S))
            if n_dirs >= 4:
                sd_no_bulk = ablate_directions(state_dict, param_keys, Vh, [2, 3])
                model.load_state_dict(sd_no_bulk)
                model.eval()
                with torch.no_grad():
                    nb_acc = masked_accuracy(model(test_src, test_tgt_in,
                        src_pad_mask=(test_src==pad_id), tgt_pad_mask=(test_tgt_in==pad_id)), test_tgt_out)
                bulk_delta = nb_acc - base_acc
            else:
                bulk_delta = 0

            print(f"    Ablate edge: Δacc={edge_delta:+.3f}, bulk: Δacc={bulk_delta:+.3f}")

            # ── Hessian curvature ──
            hessians = []
            for k in range(min(4, len(S))):
                curv, _ = hessian_curvature(build_scan_model, cfg, ckpt, state_dict,
                                             param_keys, Vh[k], test_src[:50],
                                             test_tgt_in[:50], test_tgt_out[:50])
                hessians.append(float(curv))
                label = "EDGE" if k < 2 else "BULK"
                print(f"    v{k+1} ({label}): H_curv={curv:.4f}, σ={S[k]:.4f}")

            # ── Nonlinear probes (decoder output) ──
            model.load_state_dict(state_dict)
            model.eval()
            dec_reps = {}
            def hook_fn(module, inp, out):
                dec_reps["dec"] = out.detach().cpu().numpy()
            hook = model.transformer.decoder.layers[-1].register_forward_hook(hook_fn)
            with torch.no_grad():
                model(test_src, test_tgt_in,
                      src_pad_mask=(test_src==pad_id), tgt_pad_mask=(test_tgt_in==pad_id))
            hook.remove()

            R = dec_reps.get("dec")
            r2_lin = r2_mlp = 0
            if R is not None:
                N, T, D = R.shape
                flat = R.reshape(-1, D)
                t_flat = tgt_tokens.reshape(-1)
                m_flat = tgt_mask.reshape(-1).astype(bool)
                X_p = flat[m_flat]
                y_p = t_flat[m_flat]
                n = len(X_p)
                perm = np.random.permutation(n)
                s = int(0.7 * n)

                probe = Ridge(alpha=1.0)
                probe.fit(X_p[perm[:s]], y_p[perm[:s]])
                r2_lin = r2_score(y_p[perm[s:]], probe.predict(X_p[perm[s:]]))

                mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500,
                                    early_stopping=True, random_state=42, alpha=0.01)
                try:
                    mlp.fit(X_p[perm[:s]], y_p[perm[:s]])
                    r2_mlp = r2_score(y_p[perm[s:]], mlp.predict(X_p[perm[s:]]))
                except Exception:
                    pass

            print(f"    Probes: linear={r2_lin:.3f}, mlp={r2_mlp:.3f}")

            # ── Random direction control (10 trials) ──
            rand_deltas = []
            for trial in range(10):
                rng = np.random.RandomState(trial + 3000)
                p_dim = sum(n for _, n in param_keys)
                raw = rng.randn(2, p_dim)
                Q, _ = np.linalg.qr(raw.T)
                rand_dirs = Q[:, :2].T

                theta = get_attn_param_vector(state_dict).numpy()
                removal = np.zeros_like(theta)
                for ki in range(2):
                    removal += np.dot(theta, rand_dirs[ki]) * rand_dirs[ki]

                sd_rand = {k: v.clone() for k, v in state_dict.items()}
                offset = 0
                for key, numel in param_keys:
                    chunk = removal[offset:offset + numel]
                    sd_rand[key] = sd_rand[key] - torch.tensor(chunk, dtype=sd_rand[key].dtype).reshape(sd_rand[key].shape)
                    offset += numel

                model.load_state_dict(sd_rand)
                model.eval()
                with torch.no_grad():
                    r_acc = masked_accuracy(model(test_src, test_tgt_in,
                        src_pad_mask=(test_src==pad_id), tgt_pad_mask=(test_tgt_in==pad_id)), test_tgt_out)
                rand_deltas.append(r_acc - base_acc)

            rand_deltas = np.array(rand_deltas)
            print(f"    Random ablation: mean Δacc={rand_deltas.mean():+.3f} ± {rand_deltas.std():.3f}")
            edge_ratio = abs(edge_delta) / (abs(rand_deltas.mean()) + 1e-8)
            print(f"    Edge is {edge_ratio:.1f}x more impactful")

            all_results[tag][phase_name] = {
                "step": step, "base_acc": float(base_acc),
                "edge_delta": float(edge_delta), "bulk_delta": float(bulk_delta),
                "hessians": hessians,
                "r2_linear": float(r2_lin), "r2_mlp": float(r2_mlp),
                "random_mean": float(rand_deltas.mean()),
                "random_std": float(rand_deltas.std()),
                "edge_over_random": float(edge_ratio),
                "sigma": S[:4].tolist(),
            }

    # ── Plots ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("SCAN Full Experiment Suite", fontsize=14)

    for col, (key, ylabel) in enumerate([
        ("edge_delta", "Edge ablation Δacc"),
        ("r2_linear", "Linear probe R²"),
        ("r2_mlp", "MLP probe R²"),
    ]):
        for row, tag in enumerate(["grok", "memo"]):
            if tag not in all_results:
                continue
            ax = axes[row, col]
            vals = [all_results[tag].get(p, {}).get(key, 0) for p in phases]
            color = "steelblue" if tag == "grok" else "coral"
            ax.bar(range(len(phases)), vals, color=color, alpha=0.8)
            ax.set_xticks(range(len(phases)))
            ax.set_xticklabels(phases, fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{tag}: {ylabel}", fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "scan_full_suite.png", dpi=150)
    plt.close(fig)

    torch.save(all_results, FIG_DIR / "scan_full_suite_results.pt")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: SCAN Full Suite")
    print("="*70)
    for tag in ["grok", "memo"]:
        if tag not in all_results:
            continue
        print(f"\n  {tag}:")
        for phase in phases:
            r = all_results[tag].get(phase, {})
            if not r:
                continue
            print(f"    {phase} (step {r['step']}): "
                  f"abl_edge={r['edge_delta']:+.3f}, "
                  f"H_curv={r['hessians']}, "
                  f"lin={r['r2_linear']:.3f}, mlp={r['r2_mlp']:.3f}, "
                  f"edge/random={r['edge_over_random']:.1f}x")


if __name__ == "__main__":
    main()
