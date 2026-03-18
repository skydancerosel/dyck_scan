#!/usr/bin/env python3
"""
Causal intervention experiment on Dyck depth-prediction grokking.

Replicates the intervention framework from the integrability-grokking paper
on the Dyck-1 balanced-parentheses depth-prediction task.

Two experimental arms:
  1A (Induce):  Artificially boost defect -> does grokking accelerate?
  1B (Suppress): Suppress defect -> does grokking delay/fail?

Two-phase design:
  Phase 1: Train baseline, extract PCA basis B from trajectory
  Phase 2: Re-train with interventions using fixed B from Phase 1

Conditions:
  baseline   -- Normal training
  1A-kick    -- One-time weight perturbation along defect eigenvector
  1A-noise   -- Repeated orthogonal noise injection
  1B-project -- Project out orthogonal gradient component
  1B-penalty -- Scale down orthogonal gradient component

Produces:
  figI1 -- Defect trajectories per condition
  figI2 -- Test accuracy overlay (hero figure)
  figI3 -- Grok step comparison (bar chart)
  figI5 -- Hyperparameter sensitivity
"""

import math, time, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from dyck_grok_sweep import (
    DyckSweepConfig, DyckTransformerLM, make_batch, masked_ce_loss,
    VOCAB_SIZE, TOK_PAD, get_device, eval_loss_acc,
    build_depth_dataset, split_dataset, sample_batch, eval_on_dataset,
    extract_attn_matrices, flatten_model_params,
)
from dyck_commutator_analysis import (
    _param_offsets, _write_params,
    commutator_defect, commutator_defect_median, build_pca_basis,
)
from dyck_generalization_dynamics import find_spike_step, find_grok_step_from_records

# -- config -------------------------------------------------------------------
OUT_DIR = Path(__file__).parent / "dyck_pca_plots"
SEEDS = [42, 137, 2024]

COMM_EVERY = 100
COMM_K = 5
COMM_ETA = 1e-3
MAX_STEPS = 20_000
POST_GROK_STEPS = 1000

T_START = 500
SWEEP_MAX_STEPS = 20_000

PRIMARY_HPARAMS = {
    "baseline":    {},
    "1A-kick":     {"alpha": 10, "t_start": T_START},
    "1A-noise":    {"epsilon": 0.1, "noise_interval": 50, "t_start": T_START},
    "1B-project":  {"strength": 0.5, "t_start": T_START},
    "1B-penalty":  {"lambda_penalty": 0.5, "penalty_interval": 10, "t_start": T_START},
}
CONDITIONS = list(PRIMARY_HPARAMS.keys())

SWEEP_GRIDS = {
    "1B-project": [{"strength": s, "t_start": T_START} for s in [0.25, 0.5, 0.75, 1.0]],
}

CONDITION_COLORS = {
    "baseline":    "#333333",
    "1A-kick":     "#e74c3c",
    "1A-noise":    "#e67e22",
    "1B-project":  "#2980b9",
    "1B-penalty":  "#8e44ad",
}
CONDITION_LABELS = {
    "baseline":    "Baseline",
    "1A-kick":     "1A: Defect kick",
    "1A-noise":    "1A: Orthogonal noise",
    "1B-project":  "1B: Gradient projection",
    "1B-penalty":  "1B: Gradient penalty",
}


# ==============================================================================
# Intervention functions
# ==============================================================================

def apply_defect_kick(model, batch_fn, device, B, hparams):
    alpha = hparams.get("alpha", 10)
    out = commutator_defect_median(model, batch_fn, device, K=9, eta=COMM_ETA)
    delta = out["median_delta"].to(device)
    _, _, _, normA, normB = commutator_defect(model, batch_fn, device, eta=COMM_ETA)
    grad_step_norm = normA.item()

    proj_coeffs = B.T @ delta
    delta_proj = B @ proj_coeffs
    delta_perp = delta - delta_proj
    perp_norm = delta_perp.norm()
    if perp_norm < 1e-15:
        return
    direction = delta_perp / perp_norm
    epsilon = alpha * grad_step_norm
    theta = flatten_model_params(model).to(device)
    _write_params(model, theta + epsilon * direction)
    print(f"    KICK applied: alpha={alpha}, epsilon={epsilon:.4f}")


def inject_orthogonal_noise(model, B, hparams, rng):
    epsilon = hparams.get("epsilon", 0.1)
    device = next(model.parameters()).device
    theta = flatten_model_params(model).to(device)
    noise = torch.randn(theta.shape[0], generator=rng, device=device)
    noise_proj = B @ (B.T @ noise)
    noise_perp = noise - noise_proj
    n_norm = noise_perp.norm()
    if n_norm < 1e-15:
        return
    noise_perp = noise_perp / n_norm * epsilon
    _write_params(model, theta + noise_perp)


def project_gradient_to_pca(model, B, strength=1.0):
    grads, params_with_grad = [], []
    for p in model.parameters():
        if not p.requires_grad or p.grad is None:
            continue
        grads.append(p.grad.flatten())
        params_with_grad.append(p)
    if not grads:
        return
    grad_flat = torch.cat(grads)
    device = grad_flat.device
    B_dev = B.to(device)
    grad_parallel = B_dev @ (B_dev.T @ grad_flat)
    grad_perp = grad_flat - grad_parallel
    grad_new = grad_flat - strength * grad_perp
    offset = 0
    for p in params_with_grad:
        n = p.grad.numel()
        p.grad.copy_(grad_new[offset:offset + n].view_as(p.grad))
        offset += n


def scale_orthogonal_gradient(model, B, hparams):
    lam = hparams.get("lambda_penalty", 0.5)
    project_gradient_to_pca(model, B, strength=lam)


def hparams_key(hparams):
    return tuple(sorted(hparams.items()))


def strict_grok_step(data):
    """Patience-based grok step from training state."""
    if data.get("grokked") and data.get("grok_step") is not None:
        return data["grok_step"]
    return None


def first_acc_cross_step(records, acc_threshold=0.99):
    """First evaluation step with test_acc above threshold."""
    for r in records:
        if r.get("test_acc", 0) >= acc_threshold:
            return r["step"]
    return None


# ==============================================================================
# Phase 1: Baseline + PCA basis
# ==============================================================================

def run_baseline(seed, checkpoint_every=200):
    """Train baseline model with checkpoints and build PCA basis."""
    cfg = DyckSweepConfig(WEIGHT_DECAY=1.0, SEED=seed)
    device = get_device()
    print(f"  Phase 1: Training baseline seed={seed}...")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build fixed dataset
    X_all, Y_all = build_depth_dataset(
        n_seqs=cfg.N_TOTAL, max_pairs=cfg.MAX_PAIRS,
        ctx_len=cfg.CTX_LEN, seed=cfg.DATA_SEED
    )
    frac = cfg.N_TRAIN / cfg.N_TOTAL
    train_x, train_y, test_x, test_y = split_dataset(
        X_all, Y_all, frac_train=frac, seed=cfg.DATA_SEED
    )

    ctx_len_max = max(cfg.CTX_LEN, cfg.CTX_LEN_OOD)
    model = DyckTransformerLM(
        vocab_size=VOCAB_SIZE, ctx_len=ctx_len_max,
        d_model=cfg.D_MODEL, n_layers=cfg.N_LAYERS,
        n_heads=cfg.N_HEADS, d_ff=cfg.D_FF, dropout=cfg.DROPOUT,
        n_classes=cfg.N_CLASSES,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    )

    attn_logs = [{"step": 0, "layers": extract_attn_matrices(model)}]
    metrics = []
    patience = 0
    grokked = False
    t0 = time.time()

    for step in range(1, cfg.STEPS + 1):
        model.train()
        bx, by = sample_batch(train_x, train_y, cfg.BATCH_SIZE, device)
        logits = model(bx)
        loss = masked_ce_loss(logits, by)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        if step % cfg.MODEL_LOG_EVERY == 0:
            attn_logs.append({"step": step, "layers": extract_attn_matrices(model)})

        if step % cfg.EVAL_EVERY == 0 or step == 1:
            train_loss, train_acc = eval_on_dataset(model, train_x, train_y, device)
            test_loss, test_acc = eval_on_dataset(model, test_x, test_y, device)
            metrics.append({
                "step": step,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            })

            if step % (cfg.EVAL_EVERY * 10) == 0:
                elapsed = (time.time() - t0) / 60
                print(f"    step {step:6d} | train {train_loss:.4f}/{train_acc:.3f} | "
                      f"test {test_loss:.4f}/{test_acc:.3f} | {elapsed:.1f}m")

            if test_acc >= cfg.STOP_ACC:
                patience += 1
                if patience >= cfg.STOP_PATIENCE:
                    grokked = True
                    print(f"    GROKKED at step {step} (test_acc={test_acc:.4f})")
                    break
            else:
                patience = 0

    print(f"    grokked={grokked}, {len(attn_logs)} attn snapshots")

    print(f"    Building PCA basis...")
    B = build_pca_basis(model, attn_logs, n_components=2, device="cpu")
    if B is not None:
        print(f"    Basis shape: {B.shape}")

    grok_step = None
    for m in metrics:
        if m["test_acc"] >= cfg.STOP_ACC:
            grok_step = m["step"]
            break

    return B, {"grokked": grokked, "grok_step": grok_step, "metrics": metrics}


# ==============================================================================
# Phase 2: Intervention training loop
# ==============================================================================

def train_with_intervention(wd, seed, condition, B, hparams, max_steps=None):
    device = get_device()
    steps = max_steps if max_steps is not None else MAX_STEPS
    cfg = DyckSweepConfig(WEIGHT_DECAY=wd, SEED=seed, STEPS=steps)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build fixed dataset
    X_all, Y_all = build_depth_dataset(
        n_seqs=cfg.N_TOTAL, max_pairs=cfg.MAX_PAIRS,
        ctx_len=cfg.CTX_LEN, seed=cfg.DATA_SEED
    )
    frac = cfg.N_TRAIN / cfg.N_TOTAL
    train_x, train_y, test_x, test_y = split_dataset(
        X_all, Y_all, frac_train=frac, seed=cfg.DATA_SEED
    )

    ctx_len_max = max(cfg.CTX_LEN, cfg.CTX_LEN_OOD)
    model = DyckTransformerLM(
        vocab_size=VOCAB_SIZE, ctx_len=ctx_len_max,
        d_model=cfg.D_MODEL, n_layers=cfg.N_LAYERS,
        n_heads=cfg.N_HEADS, d_ff=cfg.D_FF, dropout=cfg.DROPOUT,
        n_classes=cfg.N_CLASSES,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=wd,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    )
    B_dev = B.to(device) if B is not None else None

    def batch_fn():
        return sample_batch(train_x, train_y, cfg.BATCH_SIZE, device)

    intervention_rng = torch.Generator(device=device)
    intervention_rng.manual_seed(seed + 99999)

    t_start = hparams.get("t_start", T_START)
    kick_applied = False
    records = []
    grokked = False
    grok_step = None
    patience = 0
    steps_after_grok = 0
    t0 = time.time()

    # Step 0 eval
    test_loss, test_acc = eval_on_dataset(model, test_x, test_y, device)
    defects = []
    for _ in range(COMM_K):
        D, delta, gcos, nA, nB = commutator_defect(model, batch_fn, device, eta=COMM_ETA)
        defects.append(D)
    records.append({
        "step": 0,
        "defect_median": float(np.median(defects)),
        "defect_p25": float(np.percentile(defects, 25)),
        "defect_p75": float(np.percentile(defects, 75)),
        "test_loss": test_loss,
        "test_acc": test_acc,
    })

    for step in range(1, cfg.STEPS + 1):
        model.train()
        bx, by = sample_batch(train_x, train_y, cfg.BATCH_SIZE, device)
        logits = model(bx)
        loss = masked_ce_loss(logits, by)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)

        # Interventions
        if step >= t_start and B_dev is not None:
            if condition == "1A-kick" and not kick_applied:
                apply_defect_kick(model, batch_fn, device, B_dev, hparams)
                kick_applied = True
            elif condition == "1A-noise":
                noise_interval = hparams.get("noise_interval", 50)
                if step % noise_interval == 0:
                    inject_orthogonal_noise(model, B_dev, hparams, intervention_rng)
            elif condition == "1B-project":
                project_gradient_to_pca(model, B_dev, hparams.get("strength", 0.5))
            elif condition == "1B-penalty":
                penalty_interval = hparams.get("penalty_interval", 10)
                if step % penalty_interval == 0:
                    scale_orthogonal_gradient(model, B_dev, hparams)

        opt.step()

        if step % COMM_EVERY == 0:
            model.eval()
            test_loss, test_acc = eval_on_dataset(model, test_x, test_y, device)
            defects = []
            for _ in range(COMM_K):
                D, delta, gcos, nA, nB = commutator_defect(
                    model, batch_fn, device, eta=COMM_ETA)
                defects.append(D)
            records.append({
                "step": step,
                "defect_median": float(np.median(defects)),
                "defect_p25": float(np.percentile(defects, 25)),
                "defect_p75": float(np.percentile(defects, 75)),
                "test_loss": test_loss,
                "test_acc": test_acc,
            })
            model.train()

        if step % cfg.EVAL_EVERY == 0:
            if step % COMM_EVERY != 0:
                test_loss, test_acc = eval_on_dataset(model, test_x, test_y, device)
            if test_acc >= cfg.STOP_ACC:
                patience += 1
                if patience >= cfg.STOP_PATIENCE and not grokked:
                    grokked = True
                    grok_step = step
                    print(f"      GROKKED at step {step} (test_acc={test_acc:.4f})")
            else:
                patience = 0

        if grokked:
            steps_after_grok += 1
            if steps_after_grok >= POST_GROK_STEPS:
                break

        if step % 2000 == 0:
            elapsed = (time.time() - t0) / 60
            last_r = records[-1] if records else {}
            d = last_r.get("defect_median", 0)
            ta = last_r.get("test_acc", 0)
            print(f"      step {step:6d} | test_acc {ta:.4f} | defect {d:.1f} | "
                  f"{elapsed:.1f}m")

    return {
        "records": records,
        "grokked": grokked,
        "grok_step": grok_step,
        "condition": condition,
        "hparams": hparams,
        "seed": seed,
    }


# ==============================================================================
# Figures
# ==============================================================================

def make_figI1(all_runs, out_dir):
    """Defect trajectories per condition."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    for idx, cond in enumerate(CONDITIONS):
        ax = axes[idx // 2, idx % 2]
        ax2 = ax.twinx()
        for seed in SEEDS:
            key = (cond, seed, hparams_key(PRIMARY_HPARAMS[cond]))
            if key not in all_runs:
                continue
            data = all_runs[key]
            recs = data["records"]
            if not recs:
                continue
            steps = [r["step"] for r in recs]
            defects = [r["defect_median"] for r in recs]
            test_accs = [r["test_acc"] for r in recs]
            color = CONDITION_COLORS[cond]
            alpha_val = 0.5 + 0.2 * SEEDS.index(seed)
            ax.plot(steps, defects, color=color, linewidth=1.5, alpha=alpha_val)
            ax2.plot(steps, test_accs, color=color, linewidth=1.2,
                     linestyle="--", alpha=alpha_val * 0.7)
        if cond != "baseline":
            ax.axvline(x=T_START, color="gray", linestyle="-.", alpha=0.5, linewidth=1.5)
        ax.set_yscale("log")
        ax.set_ylabel("Defect")
        ax2.set_ylabel("Test accuracy", color="#666")
        ax.set_xlabel("Step")
        ax.set_title(CONDITION_LABELS[cond], fontsize=12, color=CONDITION_COLORS[cond])
        ax.grid(alpha=0.2)

    if len(CONDITIONS) < 6:
        axes[2, 1].set_visible(False)
    fig.suptitle("Dyck Depth Prediction: Commutator Defect Under Causal Interventions",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "figI1_dyck_intervention_defect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figI1_dyck_intervention_defect.png")


def make_figI2(all_runs, out_dir):
    """Test accuracy overlay -- all conditions."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for cond in CONDITIONS:
        all_steps = set()
        seed_data = {}
        for seed in SEEDS:
            key = (cond, seed, hparams_key(PRIMARY_HPARAMS[cond]))
            if key not in all_runs:
                continue
            recs = all_runs[key]["records"]
            if not recs:
                continue
            sd = {r["step"]: r["test_acc"] for r in recs}
            seed_data[seed] = sd
            all_steps.update(sd.keys())
        if not seed_data:
            continue
        steps_sorted = sorted(all_steps)
        means, lows, highs = [], [], []
        for s in steps_sorted:
            vals = [sd[s] for sd in seed_data.values() if s in sd]
            if vals:
                means.append(np.mean(vals))
                lows.append(np.min(vals))
                highs.append(np.max(vals))
            else:
                means.append(float("nan"))
                lows.append(float("nan"))
                highs.append(float("nan"))
        color = CONDITION_COLORS[cond]
        ax.plot(steps_sorted, means, color=color, linewidth=2.5,
                label=CONDITION_LABELS[cond])
        ax.fill_between(steps_sorted, lows, highs, color=color, alpha=0.15)

    ax.axvline(x=T_START, color="gray", linestyle="-.", alpha=0.6, linewidth=1.5)
    ax.axhline(y=0.99, color="green", linestyle=":", alpha=0.4, linewidth=1.0,
               label="Grok threshold (0.99)")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.2)
    ax.set_title("Dyck Depth Prediction: Grokking Under Causal Interventions\n"
                 "(mean +/- range over 3 seeds, wd=1.0)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "figI2_dyck_intervention_acc_overlay.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figI2_dyck_intervention_acc_overlay.png")


def make_figI3(all_runs, out_dir):
    """Grok timing comparison: strict vs first-acc-crossing."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(CONDITIONS))
    width = 0.36
    strict_steps_by_cond = {}
    acc_steps_by_cond = {}

    for cond in CONDITIONS:
        strict_steps = []
        acc_steps = []
        for seed in SEEDS:
            key = (cond, seed, hparams_key(PRIMARY_HPARAMS[cond]))
            if key not in all_runs:
                continue
            data = all_runs[key]
            strict_step = strict_grok_step(data)
            if strict_step is not None:
                strict_steps.append(strict_step)
            acc_step = first_acc_cross_step(data["records"], acc_threshold=0.99)
            if acc_step is not None:
                acc_steps.append(acc_step)
        strict_steps_by_cond[cond] = strict_steps
        acc_steps_by_cond[cond] = acc_steps

    strict_means, strict_stds = [], []
    acc_means, acc_stds = [], []
    for cond in CONDITIONS:
        strict_vals = strict_steps_by_cond[cond]
        acc_vals = acc_steps_by_cond[cond]
        strict_means.append(np.mean(strict_vals) if strict_vals else MAX_STEPS)
        strict_stds.append(np.std(strict_vals) if len(strict_vals) > 1 else 0)
        acc_means.append(np.mean(acc_vals) if acc_vals else MAX_STEPS)
        acc_stds.append(np.std(acc_vals) if len(acc_vals) > 1 else 0)

    ax.bar(x_pos - width / 2, strict_means, width, yerr=strict_stds,
           color="#1f77b4", alpha=0.85, capsize=4, edgecolor="k", linewidth=0.5,
           label="Strict grok (patience)")
    ax.bar(x_pos + width / 2, acc_means, width, yerr=acc_stds,
           color="#ff7f0e", alpha=0.85, capsize=4, edgecolor="k", linewidth=0.5,
           label="First acc>=0.99")

    for i, cond in enumerate(CONDITIONS):
        strict_vals = strict_steps_by_cond[cond]
        acc_vals = acc_steps_by_cond[cond]
        ax.text(i - width / 2, strict_means[i] + strict_stds[i] + 120,
                f"{len(strict_vals)}/{len(SEEDS)}", ha="center", fontsize=8, color="#1f77b4")
        ax.text(i + width / 2, acc_means[i] + acc_stds[i] + 120,
                f"{len(acc_vals)}/{len(SEEDS)}", ha="center", fontsize=8, color="#ff7f0e")
        for j, v in enumerate(strict_vals):
            ax.scatter(i - width / 2 + (j - 1) * 0.05, v, color="black", s=18,
                       zorder=5, alpha=0.6)
        for j, v in enumerate(acc_vals):
            ax.scatter(i + width / 2 + (j - 1) * 0.05, v, color="black", s=18,
                       zorder=5, alpha=0.6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS],
                       fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, MAX_STEPS * 1.08)
    ax.set_ylabel("Step")
    ax.set_title("Dyck Depth Prediction: Timing Under Intervention\n"
                 "(strict vs first acc>=0.99)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "figI3_dyck_intervention_timing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figI3_dyck_intervention_timing.png")


def make_figI5(all_runs, sweep_runs, out_dir):
    """Dose-response for 1B-project: strict vs first-acc-crossing."""
    fig, ax = plt.subplots(figsize=(8, 5))
    grid = SWEEP_GRIDS["1B-project"]
    param_vals = [hp["strength"] for hp in grid]
    strict_means, strict_all = [], []
    acc_means, acc_all = [], []

    for hp in grid:
        strict_steps = []
        acc_steps = []
        for seed in SEEDS:
            key = ("1B-project", seed, hparams_key(hp))
            if key in sweep_runs:
                data = sweep_runs[key]
                strict_step = strict_grok_step(data)
                if strict_step is not None:
                    strict_steps.append(strict_step)
                acc_step = first_acc_cross_step(data["records"], acc_threshold=0.99)
                if acc_step is not None:
                    acc_steps.append(acc_step)
        strict_all.append(strict_steps)
        acc_all.append(acc_steps)
        strict_means.append(np.mean(strict_steps) if strict_steps else MAX_STEPS)
        acc_means.append(np.mean(acc_steps) if acc_steps else MAX_STEPS)

    # Baseline references
    baseline_strict_steps = []
    baseline_acc_steps = []
    for seed in SEEDS:
        key = ("baseline", seed, hparams_key(PRIMARY_HPARAMS["baseline"]))
        if key in all_runs:
            data = all_runs[key]
            strict_step = strict_grok_step(data)
            if strict_step is not None:
                baseline_strict_steps.append(strict_step)
            acc_step = first_acc_cross_step(data["records"], acc_threshold=0.99)
            if acc_step is not None:
                baseline_acc_steps.append(acc_step)
    baseline_strict_mean = np.mean(baseline_strict_steps) if baseline_strict_steps else None
    baseline_acc_mean = np.mean(baseline_acc_steps) if baseline_acc_steps else None

    ax.plot(param_vals, strict_means, "o-", color="#1f77b4", linewidth=2, markersize=8,
            label="Strict grok")
    ax.plot(param_vals, acc_means, "s-", color="#ff7f0e", linewidth=2, markersize=7,
            label="First acc>=0.99")
    for i, vals in enumerate(strict_all):
        for v in vals:
            ax.scatter(param_vals[i], v, color="#1f77b4", s=20, alpha=0.5)
    for i, vals in enumerate(acc_all):
        for v in vals:
            ax.scatter(param_vals[i], v, color="#ff7f0e", s=20, alpha=0.5)
    if baseline_strict_mean is not None:
        ax.axhline(y=baseline_strict_mean, color="#1f77b4", linestyle="--", alpha=0.5,
                   linewidth=1.2, label="Baseline strict")
    if baseline_acc_mean is not None:
        ax.axhline(y=baseline_acc_mean, color="#ff7f0e", linestyle="--", alpha=0.5,
                   linewidth=1.2, label="Baseline acc>=0.99")

    ax.set_xlabel("Suppression strength", fontsize=11)
    ax.set_ylabel("Step", fontsize=11)
    ax.set_ylim(0, SWEEP_MAX_STEPS * 1.08)
    ax.set_title("Dyck Depth Prediction: Dose-Response for Gradient Projection\n"
                 "(strict vs first acc>=0.99)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figI5_dyck_dose_response.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figI5_dyck_dose_response.png")


# ==============================================================================
# Main
# ==============================================================================

def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    cache_path = OUT_DIR / "dyck_intervention_results.pt"
    all_runs = {}
    pca_bases = {}

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        all_runs = cached.get("all_runs", {})
        pca_bases = cached.get("pca_bases", {})
        print(f"  Loaded {len(all_runs)} cached runs")

    # Phase 1: Baseline + PCA
    print(f"\n{'='*70}")
    print("  PHASE 1: Baseline Training + PCA Basis")
    print(f"{'='*70}")
    for seed in SEEDS:
        if seed in pca_bases:
            print(f"\n  seed={seed}: cached")
            continue
        print(f"\n  seed={seed}:")
        B, baseline_data = run_baseline(seed)
        pca_bases[seed] = B
        torch.save({"all_runs": all_runs, "pca_bases": pca_bases}, cache_path)

    # Phase 2: Primary interventions
    print(f"\n{'='*70}")
    print("  PHASE 2: Primary Intervention Runs")
    print(f"{'='*70}")

    total = len(CONDITIONS) * len(SEEDS)
    run_i = 0
    for cond in CONDITIONS:
        for seed in SEEDS:
            run_i += 1
            hp = PRIMARY_HPARAMS[cond]
            key = (cond, seed, hparams_key(hp))
            if key in all_runs:
                data = all_runs[key]
                print(f"\n  [{run_i}/{total}] {cond} s={seed} -- cached "
                      f"(grokked={data['grokked']})")
                continue
            print(f"\n  [{run_i}/{total}] {cond} s={seed}")
            B = pca_bases.get(seed)
            data = train_with_intervention(1.0, seed, cond, B, hp)
            all_runs[key] = data
            print(f"    -> grokked={data['grokked']} (step={data['grok_step']})")
            torch.save({"all_runs": all_runs, "pca_bases": pca_bases}, cache_path)

    # Phase 2b: Sweep for dose-response
    print(f"\n{'='*70}")
    print("  PHASE 2b: Dose-Response Sweep (1B-project)")
    print(f"{'='*70}")
    sweep_runs = dict(all_runs)
    for hp in SWEEP_GRIDS["1B-project"]:
        for seed in SEEDS:
            key = ("1B-project", seed, hparams_key(hp))
            if key in sweep_runs:
                print(f"  1B-project {hp} s={seed} -- cached")
                continue
            print(f"  1B-project {hp} s={seed}")
            B = pca_bases.get(seed)
            data = train_with_intervention(1.0, seed, "1B-project", B, hp,
                                          max_steps=SWEEP_MAX_STEPS)
            sweep_runs[key] = data
            print(f"    -> grokked={data['grokked']} (step={data['grok_step']})")
            torch.save({"all_runs": all_runs, "sweep_runs": sweep_runs,
                        "pca_bases": pca_bases}, cache_path)

    # Summary
    print(f"\n{'='*80}")
    print("  INTERVENTION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Condition':>20s}  {'Strict':>6s}  {'StrictStep':>10s}  "
          f"{'AccStep':>8s}  {'Spike':>8s}  {'LeadStrict':>10s}  {'LeadAcc':>8s}")
    for cond in CONDITIONS:
        for seed in SEEDS:
            key = (cond, seed, hparams_key(PRIMARY_HPARAMS[cond]))
            if key not in all_runs:
                continue
            data = all_runs[key]
            recs = data["records"]
            spike = find_spike_step(recs)
            tag = f"{cond} s={seed}"
            strict_step = strict_grok_step(data)
            acc_step = first_acc_cross_step(recs, acc_threshold=0.99)
            lead_strict = (strict_step - spike) if (strict_step is not None and spike is not None) else None
            lead_acc = (acc_step - spike) if (acc_step is not None and spike is not None) else None
            strict_flag = "YES" if strict_step is not None else "no"
            strict_str = str(strict_step) if strict_step is not None else "---"
            acc_str = str(acc_step) if acc_step is not None else "---"
            spike_str = str(spike) if spike is not None else "---"
            lead_strict_str = str(lead_strict) if lead_strict is not None else "---"
            lead_acc_str = str(lead_acc) if lead_acc is not None else "---"
            print(f"  {tag:>20s}  {strict_flag:>6s}  {strict_str:>10s}  "
                  f"{acc_str:>8s}  {spike_str:>8s}  {lead_strict_str:>10s}  {lead_acc_str:>8s}")

    # Figures
    print(f"\n  Generating figures...")
    make_figI1(all_runs, OUT_DIR)
    make_figI2(all_runs, OUT_DIR)
    make_figI3(all_runs, OUT_DIR)
    make_figI5(all_runs, sweep_runs, OUT_DIR)

    torch.save({"all_runs": all_runs, "sweep_runs": sweep_runs,
                "pca_bases": pca_bases}, cache_path)
    print(f"\n  saved {cache_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
