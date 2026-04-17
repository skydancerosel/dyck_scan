#!/usr/bin/env python3
"""
Functional perturbation curves: ε-sweep along edge vs bulk directions.

For each Gram direction v_k, sweep ε from -0.05 to +0.05 and measure:
  - validation loss
  - task accuracy
  - output KL divergence from base model
  - attention entropy change
  - commutator norm change

Compares edge (v1,v2) vs bulk (v3,v4) at pre/at/post grok.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dyck.grok_sweep import (
    DyckTransformerLM, VOCAB_SIZE, build_depth_dataset, split_dataset,
    masked_ce_loss, masked_accuracy,
)
from spectral.gram_edge_functional_modes import (
    get_attn_param_vector, get_attn_param_keys, compute_gram_svd,
)

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"

# ε sweep parameters
EPS_RANGE = np.linspace(-0.03, 0.03, 25)


def perturb_and_evaluate(model_cls, state_dict, cfg, direction_vec, param_keys,
                          eps, test_x, test_y):
    """Perturb model along direction and evaluate.

    Returns loss, accuracy, logits, and attention entropy.
    """
    # Build perturbed state dict
    perturbed_sd = {k: v.clone() for k, v in state_dict.items()}
    offset = 0
    for key, numel in param_keys:
        chunk = direction_vec[offset:offset + numel]
        perturbed_sd[key] = perturbed_sd[key] + eps * torch.tensor(
            chunk, dtype=perturbed_sd[key].dtype
        ).reshape(perturbed_sd[key].shape)
        offset += numel

    # Build model
    model = DyckTransformerLM(
        vocab_size=VOCAB_SIZE,
        ctx_len=max(cfg["CTX_LEN"], cfg["CTX_LEN_OOD"]),
        d_model=cfg["D_MODEL"], n_layers=cfg["N_LAYERS"],
        n_heads=cfg["N_HEADS"], d_ff=cfg["D_FF"],
        dropout=cfg["DROPOUT"], n_classes=cfg["N_CLASSES"],
    )
    model.load_state_dict(perturbed_sd)
    model.eval()

    with torch.no_grad():
        logits = model(test_x)
        loss = masked_ce_loss(logits, test_y).item()
        acc = masked_accuracy(logits, test_y)

    # Compute attention entropy
    attn_entropies = []
    with torch.no_grad():
        B, T = test_x.shape
        pos = torch.arange(T)
        h = model.tok_emb(test_x) + model.pos_emb(pos)[None, :, :]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T)

        for layer in model.encoder.layers:
            attn = layer.self_attn
            d = attn.embed_dim
            n_heads = attn.num_heads
            d_head = d // n_heads

            h_normed = layer.norm1(h)
            if attn._qkv_same_embed_dim:
                Wq = attn.in_proj_weight[:d]
                Wk = attn.in_proj_weight[d:2*d]
                bq = attn.in_proj_bias[:d] if attn.in_proj_bias is not None else None
                bk = attn.in_proj_bias[d:2*d] if attn.in_proj_bias is not None else None

            Q = F.linear(h_normed, Wq, bq).view(B, T, n_heads, d_head).transpose(1, 2)
            K = F.linear(h_normed, Wk, bk).view(B, T, n_heads, d_head).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)
            scores = scores + causal_mask[None, None, :, :]
            weights = torch.softmax(scores, dim=-1)

            # Entropy
            ent = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean()
            attn_entropies.append(ent.item())

            h = layer(h, src_mask=causal_mask, is_causal=True)

    # Commutator norm
    comm_norms = []
    for layer in model.encoder.layers:
        attn = layer.self_attn
        d = attn.embed_dim
        if attn._qkv_same_embed_dim:
            Wq = attn.in_proj_weight[:d].detach().numpy()
            Wk = attn.in_proj_weight[d:2*d].detach().numpy()
        comm = Wq @ Wk - Wk @ Wq
        comm_norms.append(float(np.linalg.norm(comm, 'fro')))

    return {
        "loss": loss,
        "acc": acc,
        "logits": logits,
        "attn_entropy": np.mean(attn_entropies),
        "comm_norm": np.mean(comm_norms),
    }


def compute_kl(logits_base, logits_pert, targets):
    """KL divergence between base and perturbed output distributions."""
    mask = targets != -100
    if mask.sum() == 0:
        return 0.0

    B, T, V = logits_base.shape
    p_base = F.softmax(logits_base, dim=-1).view(-1, V)[mask.view(-1)]
    p_pert = F.softmax(logits_pert, dim=-1).view(-1, V)[mask.view(-1)]

    kl = F.kl_div(torch.log(p_pert + 1e-10), p_base, reduction='batchmean')
    return kl.item()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # Test data
    X_all, Y_all = build_depth_dataset(n_seqs=5050, max_pairs=12, ctx_len=24, seed=0)
    _, _, test_x, test_y = split_dataset(X_all, Y_all, frac_train=50/5050, seed=0)
    test_x = test_x[:200]
    test_y = test_y[:200]

    phase_indices = {"pre_grok": 2, "at_grok": 5, "post_grok": 14, "late": 39}
    GRAM_WINDOW = 5

    all_results = {}

    for tag, ckpt_name in [("grok", "dyck_grok_fourier.pt"), ("memo", "dyck_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        ckpt = torch.load(ckpt_path, weights_only=False)
        snapshots = ckpt["snapshots"]
        cfg = ckpt["cfg"]

        param_keys = get_attn_param_keys(snapshots[0]["state_dict"])

        print(f"\n{'='*60}")
        print(f"Perturbation curves: {tag}")
        print(f"{'='*60}")

        all_results[tag] = {}

        for phase_name, center_idx in phase_indices.items():
            if center_idx >= len(snapshots):
                continue

            center_step = snapshots[center_idx]["step"]
            print(f"\n  {phase_name} (step {center_step}):")

            # Gram SVD
            gram = compute_gram_svd(snapshots, center_idx, GRAM_WINDOW)
            if gram is None:
                continue

            S = gram["singular_values"]
            Vh = gram["Vh"]

            state_dict = snapshots[center_idx]["state_dict"]

            # Base model evaluation
            base = perturb_and_evaluate(DyckTransformerLM, state_dict, cfg,
                                         np.zeros(Vh.shape[1]), param_keys,
                                         0, test_x, test_y)
            logits_base = base["logits"]

            phase_results = {"step": center_step, "directions": {}}

            for k in range(min(4, len(S))):
                v_k = Vh[k]
                label = "EDGE" if k < 2 else "BULK"
                print(f"    v{k+1} ({label}, σ={S[k]:.4f}): sweeping ε...", end="", flush=True)

                curves = {
                    "eps": EPS_RANGE.tolist(),
                    "loss": [], "acc": [], "kl": [],
                    "attn_entropy": [], "comm_norm": [],
                }

                for eps in EPS_RANGE:
                    result = perturb_and_evaluate(
                        DyckTransformerLM, state_dict, cfg,
                        v_k, param_keys, eps, test_x, test_y)
                    kl = compute_kl(logits_base, result["logits"], test_y)

                    curves["loss"].append(result["loss"])
                    curves["acc"].append(result["acc"])
                    curves["kl"].append(kl)
                    curves["attn_entropy"].append(result["attn_entropy"])
                    curves["comm_norm"].append(result["comm_norm"])

                # Summary: curvature (2nd derivative at ε=0)
                mid = len(EPS_RANGE) // 2
                loss_curv = (curves["loss"][mid+1] + curves["loss"][mid-1] - 2*curves["loss"][mid]) / (EPS_RANGE[1] - EPS_RANGE[0])**2
                kl_curv = (curves["kl"][mid+1] + curves["kl"][mid-1] - 2*curves["kl"][mid]) / (EPS_RANGE[1] - EPS_RANGE[0])**2
                max_kl = max(curves["kl"])

                curves["loss_curvature"] = float(loss_curv)
                curves["kl_curvature"] = float(kl_curv)
                curves["max_kl"] = float(max_kl)
                curves["sigma"] = float(S[k])

                print(f" loss_curv={loss_curv:.1f}, max_KL={max_kl:.4f}")

                phase_results["directions"][f"v{k+1}"] = curves

            all_results[tag][phase_name] = phase_results

    # ══════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════

    phases = list(phase_indices.keys())

    # ── Plot 1: Loss landscapes (4 panels × 2 models) ──
    for tag in ["grok", "memo"]:
        fig, axes = plt.subplots(2, len(phases), figsize=(5*len(phases), 8))
        fig.suptitle(f"Dyck {tag}: Loss & KL Perturbation Curves", fontsize=14)

        for col, phase in enumerate(phases):
            r = all_results[tag].get(phase, {})
            if not r:
                continue

            for k in range(min(4, len(r.get("directions", {})))):
                vname = f"v{k+1}"
                d = r["directions"].get(vname, {})
                if not d:
                    continue
                color = "steelblue" if k < 2 else "coral"
                ls = '-' if k % 2 == 0 else '--'
                label_type = "edge" if k < 2 else "bulk"

                axes[0, col].plot(d["eps"], d["loss"], f'{ls}', color=color,
                                 label=f"{vname} ({label_type})", linewidth=1.5)
                axes[1, col].plot(d["eps"], d["kl"], f'{ls}', color=color,
                                 label=f"{vname}", linewidth=1.5)

            axes[0, col].set_title(f"{phase} (step {r.get('step', '?')})", fontsize=10)
            axes[0, col].set_xlabel("ε")
            axes[1, col].set_xlabel("ε")
            if col == 0:
                axes[0, col].set_ylabel("Loss")
                axes[1, col].set_ylabel("KL divergence")
            if col == 0:
                axes[0, col].legend(fontsize=6)

        plt.tight_layout()
        fig.savefig(FIG_DIR / f"perturbation_curves_{tag}.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {FIG_DIR / f'perturbation_curves_{tag}.png'}")

    # ── Plot 2: Loss curvature comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Loss Curvature Along Edge vs Bulk Directions", fontsize=14)

    for i, tag in enumerate(["grok", "memo"]):
        ax = axes[i]
        for k, color in [(0, "steelblue"), (1, "royalblue"), (2, "coral"), (3, "lightsalmon")]:
            vals = []
            for p in phases:
                r = all_results[tag].get(p, {})
                d = r.get("directions", {}).get(f"v{k+1}", {})
                vals.append(d.get("loss_curvature", 0))
            label = "edge" if k < 2 else "bulk"
            ax.plot(range(len(phases)), vals, 'o-', color=color,
                    label=f"v{k+1} ({label})", markersize=6)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=8)
        ax.set_ylabel("∂²L/∂ε² (curvature)")
        ax.set_title(f"{tag}")
        ax.legend(fontsize=7)
        ax.set_yscale('symlog', linthresh=10)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "loss_curvature_edge_vs_bulk.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'loss_curvature_edge_vs_bulk.png'}")

    # ── Plot 3: Max KL divergence ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Max KL Divergence: Edge vs Bulk", fontsize=14)

    for i, tag in enumerate(["grok", "memo"]):
        ax = axes[i]
        for k, color in [(0, "steelblue"), (1, "royalblue"), (2, "coral"), (3, "lightsalmon")]:
            vals = []
            for p in phases:
                r = all_results[tag].get(p, {})
                d = r.get("directions", {}).get(f"v{k+1}", {})
                vals.append(d.get("max_kl", 0))
            label = "edge" if k < 2 else "bulk"
            ax.plot(range(len(phases)), vals, 'o-', color=color,
                    label=f"v{k+1} ({label})", markersize=6)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=8)
        ax.set_ylabel("max KL(base || perturbed)")
        ax.set_title(f"{tag}")
        ax.legend(fontsize=7)
        ax.set_yscale('symlog', linthresh=0.001)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "max_kl_edge_vs_bulk.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'max_kl_edge_vs_bulk.png'}")

    # ── Plot 4: Attention entropy sensitivity ──
    fig, axes = plt.subplots(2, len(phases), figsize=(5*len(phases), 8))
    fig.suptitle("Dyck Grok: Attention Entropy & Commutator Under Perturbation", fontsize=14)

    for col, phase in enumerate(phases):
        r = all_results["grok"].get(phase, {})
        if not r:
            continue
        for k in range(min(4, len(r.get("directions", {})))):
            d = r["directions"].get(f"v{k+1}", {})
            if not d:
                continue
            color = "steelblue" if k < 2 else "coral"
            ls = '-' if k % 2 == 0 else '--'
            axes[0, col].plot(d["eps"], d["attn_entropy"], f'{ls}', color=color,
                             label=f"v{k+1}", linewidth=1.5)
            axes[1, col].plot(d["eps"], d["comm_norm"], f'{ls}', color=color,
                             label=f"v{k+1}", linewidth=1.5)
        axes[0, col].set_title(phase, fontsize=10)
        axes[0, col].set_xlabel("ε")
        axes[1, col].set_xlabel("ε")
        if col == 0:
            axes[0, col].set_ylabel("Attn entropy")
            axes[1, col].set_ylabel("||[W_Q, W_K]||_F")
            axes[0, col].legend(fontsize=6)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "perturbation_entropy_commutator.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'perturbation_entropy_commutator.png'}")

    # Save
    torch.save(all_results, FIG_DIR / "perturbation_curves_results.pt")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Perturbation Curves — Edge vs Bulk")
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
                    label = "EDGE" if vname in ["v1", "v2"] else "BULK"
                    print(f"      {vname} ({label}): loss_curv={d.get('loss_curvature',0):.1f}, "
                          f"max_KL={d.get('max_kl',0):.5f}")


if __name__ == "__main__":
    main()
