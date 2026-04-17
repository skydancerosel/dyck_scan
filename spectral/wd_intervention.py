#!/usr/bin/env python3
"""
#5: Weight decay intervention — remove/vary WD after grokking.

Protocol:
  - Load model at pre-grok and post-grok checkpoints
  - Continue training with: same WD, zero WD, reduced WD, increased WD
  - Track: Gram gaps, functional perturbation, Fourier modes, entropy, probe R²

Both Dyck and SCAN.
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
from sklearn.metrics import r2_score

from dyck.grok_sweep import (
    DyckTransformerLM, DyckSweepConfig, VOCAB_SIZE,
    build_depth_dataset, split_dataset, sample_batch,
    masked_ce_loss, masked_accuracy, eval_on_dataset, get_device,
)
from spectral.gram_edge_functional_modes import get_attn_param_vector, get_attn_param_keys
from spectral.fourier_functional_dyck import extract_hidden_reps, compute_positional_power_spectrum, compute_spectral_concentration

CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"

CONTINUATION_STEPS = 3000
EVAL_EVERY = 100
LOG_EVERY = 200

WD_CONDITIONS = {
    "same_wd": 1.0,
    "zero_wd": 0.0,
    "half_wd": 0.5,
    "double_wd": 2.0,
}


def continue_training(model, cfg, train_x, train_y, test_x, test_y,
                       new_wd, steps, device):
    """Continue training from current model state with new weight decay."""
    model = model.to(device)
    model.train()

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["LR"],
        weight_decay=new_wd,
        betas=(cfg["ADAM_BETA1"], cfg["ADAM_BETA2"]),
    )

    metrics = []
    snapshots = []
    snapshots.append({"step": 0, "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()}})

    for step in range(1, steps + 1):
        model.train()
        bx, by = sample_batch(train_x, train_y, cfg["BATCH_SIZE"], device)
        logits = model(bx)
        loss = masked_ce_loss(logits, by)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["GRAD_CLIP"])
        opt.step()

        if step % EVAL_EVERY == 0:
            train_loss, train_acc = eval_on_dataset(model, train_x, train_y, device)
            test_loss, test_acc = eval_on_dataset(model, test_x, test_y, device)
            metrics.append({
                "step": step, "train_loss": train_loss, "train_acc": train_acc,
                "test_loss": test_loss, "test_acc": test_acc,
            })

        if step % LOG_EVERY == 0:
            snapshots.append({
                "step": step,
                "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            })

    return metrics, snapshots


def compute_diagnostics(model, test_x, test_y, snapshots):
    """Compute Gram gap, probe R², Fourier, attention entropy from snapshots."""
    mask = (test_y != -100).numpy()
    depths = test_y.numpy()

    diag_list = []

    for snap in snapshots:
        model.load_state_dict(snap["state_dict"])
        model.eval()

        # Hidden reps
        reps = extract_hidden_reps(model, test_x[:200])

        # Probe R²
        layer1_reps = reps.get("layer_1", reps.get("layer_0"))
        if layer1_reps is not None:
            N, T, D = layer1_reps.shape
            flat = layer1_reps.reshape(-1, D)
            d_flat = depths[:200].reshape(-1)
            m_flat = mask[:200].reshape(-1).astype(bool)
            X = flat[m_flat]
            y = d_flat[m_flat]
            n = len(X)
            if n > 50:
                perm = np.random.permutation(n)
                s = int(0.7 * n)
                probe = Ridge(alpha=1.0)
                probe.fit(X[perm[:s]], y[perm[:s]])
                pred = probe.predict(X[perm[s:]])
                r2 = r2_score(y[perm[s:]], pred)
            else:
                r2 = 0
        else:
            r2 = 0

        # Fourier concentration
        if layer1_reps is not None:
            _, power = compute_positional_power_spectrum(layer1_reps, mask[:200])
            conc = compute_spectral_concentration(power)
        else:
            conc = {"top3": 0, "dominant_omega": 0}

        # Attention entropy
        ent = 0
        with torch.no_grad():
            B, T_seq = test_x[:100].shape
            pos = torch.arange(T_seq)
            h = model.tok_emb(test_x[:100]) + model.pos_emb(pos)[None, :, :]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T_seq)
            for layer in model.encoder.layers:
                attn = layer.self_attn
                d = attn.embed_dim
                n_heads = attn.num_heads
                d_head = d // n_heads
                h_n = layer.norm1(h)
                if attn._qkv_same_embed_dim:
                    Wq = attn.in_proj_weight[:d]
                    Wk = attn.in_proj_weight[d:2*d]
                    bq = attn.in_proj_bias[:d] if attn.in_proj_bias is not None else None
                    bk = attn.in_proj_bias[d:2*d] if attn.in_proj_bias is not None else None
                Q = nn.functional.linear(h_n, Wq, bq).view(B, T_seq, n_heads, d_head).transpose(1, 2)
                K = nn.functional.linear(h_n, Wk, bk).view(B, T_seq, n_heads, d_head).transpose(1, 2)
                scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)
                scores = scores + causal_mask[None, None, :, :]
                weights = torch.softmax(scores, dim=-1)
                ent += -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean().item()
                h = layer(h, src_mask=causal_mask, is_causal=True)
            ent /= len(model.encoder.layers)

        # Param norm
        param_norm = sum(p.norm().item()**2 for p in model.parameters())**0.5

        diag_list.append({
            "step": snap["step"],
            "probe_r2": r2,
            "fourier_top3": conc["top3"],
            "dominant_omega": conc["dominant_omega"],
            "attn_entropy": ent,
            "param_norm": param_norm,
        })

    return diag_list


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    device = "cpu"  # Avoid MPS state_dict issues

    # Load grok checkpoint
    ckpt_path = CKPT_DIR / "dyck_grok_fourier.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg = ckpt["cfg"]
    train_x, train_y = ckpt["train_x"], ckpt["train_y"]
    test_x, test_y = ckpt["test_x"], ckpt["test_y"]

    # Intervention points
    intervention_points = {
        "pre_grok": 4,    # ~step 400 (before grokking at ~600)
        "post_grok": 10,  # ~step 1000 (after grokking)
    }

    all_results = {}

    for point_name, snap_idx in intervention_points.items():
        if snap_idx >= len(ckpt["snapshots"]):
            continue

        base_step = ckpt["snapshots"][snap_idx]["step"]
        base_sd = ckpt["snapshots"][snap_idx]["state_dict"]

        print(f"\n{'='*60}")
        print(f"Intervention at {point_name} (step {base_step})")
        print(f"{'='*60}")

        # Evaluate base model
        model = DyckTransformerLM(
            vocab_size=VOCAB_SIZE,
            ctx_len=max(cfg["CTX_LEN"], cfg["CTX_LEN_OOD"]),
            d_model=cfg["D_MODEL"], n_layers=cfg["N_LAYERS"],
            n_heads=cfg["N_HEADS"], d_ff=cfg["D_FF"],
            dropout=cfg["DROPOUT"], n_classes=cfg["N_CLASSES"],
        )
        model.load_state_dict(base_sd)
        model.eval()

        base_loss, base_acc = eval_on_dataset(model, test_x, test_y, device)
        print(f"  Base: test_loss={base_loss:.4f}, test_acc={base_acc:.3f}")

        point_results = {}

        for cond_name, new_wd in WD_CONDITIONS.items():
            print(f"\n  {cond_name} (wd={new_wd}):")

            # Fresh copy
            model_copy = DyckTransformerLM(
                vocab_size=VOCAB_SIZE,
                ctx_len=max(cfg["CTX_LEN"], cfg["CTX_LEN_OOD"]),
                d_model=cfg["D_MODEL"], n_layers=cfg["N_LAYERS"],
                n_heads=cfg["N_HEADS"], d_ff=cfg["D_FF"],
                dropout=cfg["DROPOUT"], n_classes=cfg["N_CLASSES"],
            )
            model_copy.load_state_dict(copy.deepcopy(base_sd))

            torch.manual_seed(42)
            random.seed(42)
            np.random.seed(42)

            # Use CPU to avoid MPS state_dict issues
            train_device = "cpu"

            metrics, snapshots = continue_training(
                model_copy, cfg, train_x, train_y, test_x, test_y,
                new_wd, CONTINUATION_STEPS, train_device
            )

            # Diagnostics at key points
            model_eval = DyckTransformerLM(
                vocab_size=VOCAB_SIZE,
                ctx_len=max(cfg["CTX_LEN"], cfg["CTX_LEN_OOD"]),
                d_model=cfg["D_MODEL"], n_layers=cfg["N_LAYERS"],
                n_heads=cfg["N_HEADS"], d_ff=cfg["D_FF"],
                dropout=cfg["DROPOUT"], n_classes=cfg["N_CLASSES"],
            )
            diagnostics = compute_diagnostics(model_eval, test_x[:300], test_y[:300], snapshots)

            final_m = metrics[-1] if metrics else {}
            final_d = diagnostics[-1] if diagnostics else {}
            print(f"    Final: test_acc={final_m.get('test_acc', 0):.3f}, "
                  f"R²={final_d.get('probe_r2', 0):.3f}, "
                  f"entropy={final_d.get('attn_entropy', 0):.2f}, "
                  f"||θ||={final_d.get('param_norm', 0):.1f}")

            point_results[cond_name] = {
                "wd": new_wd,
                "metrics": metrics,
                "diagnostics": diagnostics,
            }

        all_results[point_name] = point_results

    # ══════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════

    colors = {"same_wd": "steelblue", "zero_wd": "coral", "half_wd": "orange", "double_wd": "purple"}

    for point_name, point_results in all_results.items():
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Dyck WD Intervention from {point_name}", fontsize=14)

        plot_specs = [
            (0, 0, "test_acc", "Test Accuracy", "metrics"),
            (0, 1, "test_loss", "Test Loss", "metrics"),
            (0, 2, "param_norm", "||θ||", "diagnostics"),
            (1, 0, "probe_r2", "Depth Probe R²", "diagnostics"),
            (1, 1, "attn_entropy", "Attention Entropy", "diagnostics"),
            (1, 2, "fourier_top3", "Fourier Top-3", "diagnostics"),
        ]

        for row, col, key, ylabel, source in plot_specs:
            ax = axes[row, col]
            for cond_name, cond_data in point_results.items():
                data = cond_data[source]
                steps = [d["step"] for d in data]
                vals = [d.get(key, 0) for d in data]
                ax.plot(steps, vals, '-', color=colors[cond_name],
                        label=f"{cond_name} (wd={cond_data['wd']})", linewidth=1.5)
            ax.set_xlabel("Continuation step")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

        plt.tight_layout()
        fig.savefig(FIG_DIR / f"wd_intervention_{point_name}.png", dpi=150)
        plt.close(fig)
        print(f"\nSaved: {FIG_DIR / f'wd_intervention_{point_name}.png'}")

    # Save
    torch.save(all_results, FIG_DIR / "wd_intervention_results.pt")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Weight Decay Intervention")
    print("="*70)
    for point_name, point_results in all_results.items():
        print(f"\n  {point_name}:")
        for cond_name, cond_data in point_results.items():
            m = cond_data["metrics"][-1] if cond_data["metrics"] else {}
            d = cond_data["diagnostics"][-1] if cond_data["diagnostics"] else {}
            print(f"    {cond_name} (wd={cond_data['wd']}): "
                  f"acc={m.get('test_acc', 0):.3f}, "
                  f"R²={d.get('probe_r2', 0):.3f}, "
                  f"ent={d.get('attn_entropy', 0):.2f}, "
                  f"||θ||={d.get('param_norm', 0):.1f}")


if __name__ == "__main__":
    main()
