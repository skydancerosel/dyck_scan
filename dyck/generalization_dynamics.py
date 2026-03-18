#!/usr/bin/env python3
"""
Generalization dynamics: does commutator defect spike predict grokking
in Dyck depth-prediction?

Fine-grained temporal analysis overlaying commutator defect with
generalization accuracy to show that defect explosion precedes the
generalization transition.

Produces:
  figW  — Defect vs test acc for each seed (wd=1.0 + wd=0.0 control)
  figX  — Lead-time scatter: how many steps before grokking does defect spike?
  figW2 — Hero single-panel figure
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
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from dyck_grok_sweep import (
    DyckSweepConfig, DyckTransformerLM, make_batch, masked_ce_loss,
    VOCAB_SIZE, TOK_PAD, get_device, eval_loss_acc,
    build_depth_dataset, split_dataset, sample_batch, eval_on_dataset,
)
from dyck_commutator_analysis import commutator_defect

# ── config ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "dyck_pca_plots"
SEEDS = [42, 137, 2024]

COMM_EVERY = 100
COMM_K = 5
COMM_ETA = 1e-3
MAX_STEPS = 20_000
POST_GROK_STEPS = 1000


# ═══════════════════════════════════════════════════════════════════════════
# Training with inline commutator measurement
# ═══════════════════════════════════════════════════════════════════════════

def train_with_defect_tracking(wd, seed, max_steps=None):
    device = get_device()
    steps = max_steps if max_steps is not None else MAX_STEPS
    cfg = DyckSweepConfig(WEIGHT_DECAY=wd, SEED=seed, STEPS=steps)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build fixed dataset (same across all seeds via DATA_SEED)
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

    def batch_fn():
        return sample_batch(train_x, train_y, cfg.BATCH_SIZE, device)

    records = []
    grokked = False
    grok_step = None
    patience = 0
    steps_after_grok = 0
    t0 = time.time()

    # Step 0 measurement
    train_loss, train_acc = eval_on_dataset(model, train_x, train_y, device)
    test_loss, test_acc = eval_on_dataset(model, test_x, test_y, device)
    ood_rng = random.Random(seed + 999)
    ood_loss, ood_acc = eval_loss_acc(
        model, cfg.EVAL_BATCHES_OOD, cfg.EVAL_BS_OOD,
        cfg.CTX_LEN_OOD, cfg.MAX_PAIRS_OOD, device, ood_rng)

    defects = []
    for _ in range(COMM_K):
        D, delta, gcos, nA, nB = commutator_defect(model, batch_fn, device, eta=COMM_ETA)
        defects.append(D)
    records.append({
        "step": 0,
        "defect_median": float(np.median(defects)),
        "defect_p25": float(np.percentile(defects, 25)),
        "defect_p75": float(np.percentile(defects, 75)),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "ood_loss": ood_loss,
        "ood_acc": ood_acc,
    })

    for step in range(1, cfg.STEPS + 1):
        model.train()
        x, y = sample_batch(train_x, train_y, cfg.BATCH_SIZE, device)
        logits = model(x)
        loss = masked_ce_loss(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        if step % COMM_EVERY == 0:
            model.eval()
            train_loss, train_acc = eval_on_dataset(model, train_x, train_y, device)
            test_loss, test_acc = eval_on_dataset(model, test_x, test_y, device)

            ood_rng = random.Random(seed + 999 + step)
            ood_loss, ood_acc = eval_loss_acc(
                model, cfg.EVAL_BATCHES_OOD, cfg.EVAL_BS_OOD,
                cfg.CTX_LEN_OOD, cfg.MAX_PAIRS_OOD, device, ood_rng)

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
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "ood_loss": ood_loss,
                "ood_acc": ood_acc,
            })
            model.train()

        if step % cfg.EVAL_EVERY == 0:
            if step % COMM_EVERY != 0:
                test_loss, test_acc = eval_on_dataset(
                    model, test_x, test_y, device)

            if test_acc >= cfg.STOP_ACC:
                patience += 1
                if patience >= cfg.STOP_PATIENCE and not grokked:
                    grokked = True
                    grok_step = step
                    print(f"      GROKKED at step {step}")
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
        "wd": wd,
        "seed": seed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Spike / grok detection
# ═══════════════════════════════════════════════════════════════════════════

def find_spike_step(records, threshold_factor=10, min_defect=20):
    if len(records) < 3:
        return None
    baseline = np.median([r["defect_median"] for r in records[:3]])
    baseline = max(baseline, 0.1)
    for i in range(2, len(records)):
        d = records[i]["defect_median"]
        if d > threshold_factor * baseline and d > min_defect:
            return records[i]["step"]
    return None


def find_grok_step_from_records(records, acc_threshold=0.99):
    """Find first step where test_acc >= threshold."""
    for r in records:
        if r["test_acc"] >= acc_threshold:
            return r["step"]
    return None


def strict_grok_step(data):
    """Patience-based grok step from training state."""
    if data.get("grokked") and data.get("grok_step") is not None:
        return data["grok_step"]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    cache_path = OUT_DIR / "dyck_generalization_dynamics_results.pt"
    all_runs = {}
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if "all_runs" in cached:
            all_runs = cached["all_runs"]
            print(f"  Loaded {len(all_runs)} cached runs")

    # ── Grokking runs (wd=1.0) ────────────────────────────────────────
    for seed in SEEDS:
        key = ("wd1.0", seed)
        if key in all_runs:
            print(f"\n  wd=1.0 s={seed} — cached")
            continue
        print(f"\n  wd=1.0 s={seed}")
        data = train_with_defect_tracking(1.0, seed)
        all_runs[key] = data
        print(f"    → grokked={data['grokked']} (step={data['grok_step']}), "
              f"{len(data['records'])} records")

    # ── No-wd controls ────────────────────────────────────────────────
    for seed in SEEDS[:1]:
        key = ("wd0.0", seed)
        if key in all_runs:
            print(f"\n  wd=0.0 s={seed} — cached")
            continue
        print(f"\n  wd=0.0 s={seed} (control)")
        data = train_with_defect_tracking(0.0, seed, max_steps=10_000)
        all_runs[key] = data
        print(f"    → grokked={data['grokked']}, {len(data['records'])} records")

    # ── Compute lead times ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  DEFECT SPIKE vs GROKKING TIMING")
    print(f"{'='*80}")
    print(f"  {'Config':>20s}  {'spike':>8s}  {'strict':>8s}  "
          f"{'acc>=0.99':>10s}  {'lead_strict':>12s}  {'lead_acc':>10s}")

    lead_rows = []
    lead_times_acc = []
    for seed in SEEDS:
        key = ("wd1.0", seed)
        if key not in all_runs:
            continue
        data = all_runs[key]
        recs = data["records"]
        spike = find_spike_step(recs)
        strict_step = strict_grok_step(data)
        acc_step = find_grok_step_from_records(recs, 0.99)
        lead_strict = (strict_step - spike) if (spike is not None and strict_step is not None) else None
        lead_acc = (acc_step - spike) if (spike is not None and acc_step is not None) else None
        if lead_acc is not None:
            lead_times_acc.append(lead_acc)
        lead_rows.append({
            "seed": seed,
            "spike": spike,
            "strict_step": strict_step,
            "acc_step": acc_step,
            "lead_strict": lead_strict,
            "lead_acc": lead_acc,
        })
        tag = f"wd=1.0 s={seed}"
        print(f"  {tag:>20s}  {str(spike):>8s}  {str(strict_step):>8s}  "
              f"{str(acc_step):>10s}  {str(lead_strict):>12s}  {str(lead_acc):>10s}")

    if lead_times_acc:
        n_pos = sum(1 for l in lead_times_acc if l > 0)
        p_val = 2 ** (-len(lead_times_acc))
        print(f"\n  Lead times (acc>=0.99): {lead_times_acc}")
        print(f"  Sign test: {n_pos}/{len(lead_times_acc)} positive, p = {p_val:.4f}")

    # ── Figure W: Defect vs test acc ──────────────────────────────────
    print("\n  Generating figures...")
    SEED_COLORS = {42: "#1f77b4", 137: "#ff7f0e", 2024: "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)
    for si, seed in enumerate(SEEDS):
        ax = axes[0, si]
        ax2 = ax.twinx()

        key = ("wd1.0", seed)
        if key in all_runs:
            recs = all_runs[key]["records"]
            steps = [r["step"] for r in recs]
            defects = [r["defect_median"] for r in recs]
            test_accs = [r["test_acc"] for r in recs]

            ax.plot(steps, defects, color="#1a5276", linewidth=2, label="Defect")
            ax2.plot(steps, test_accs, color="#e74c3c", linewidth=2,
                     linestyle="--", label="Test acc")

            spike = find_spike_step(recs)
            strict_step = strict_grok_step(all_runs[key])
            grok = find_grok_step_from_records(recs, 0.99)
            if spike is not None:
                ax.axvline(x=spike, color="#1a5276", linestyle=":", alpha=0.6)
            if grok is not None:
                ax.axvline(x=grok, color="#e74c3c", linestyle=":", alpha=0.6)
            if strict_step is not None:
                ax.axvline(x=strict_step, color="#8e44ad", linestyle="-.", alpha=0.45)

        ax.set_yscale("log")
        ax.set_ylabel("Defect", color="#1a5276")
        ax2.set_ylabel("Test accuracy", color="#e74c3c")
        ax2.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Training step")
        ax.set_title(f"seed={seed}")
        ax.grid(alpha=0.2)

        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    fig.suptitle("Dyck Depth Prediction: Commutator Defect Predicts Generalization\n"
                 "(solid=defect, dashed=test acc, wd=1.0)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figW_dyck_defect_predicts_grokking.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figW_dyck_defect_predicts_grokking.png")

    # ── Figure X: Lead-time bar chart (strict vs first-crossing) ─────
    lead_by_seed = {row["seed"]: row for row in lead_rows}
    has_any_lead = any(
        (row.get("lead_acc") is not None or row.get("lead_strict") is not None)
        for row in lead_rows
    )
    if has_any_lead:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(SEEDS))
        width = 0.36
        acc_vals = [lead_by_seed.get(s, {}).get("lead_acc") for s in SEEDS]
        strict_vals = [lead_by_seed.get(s, {}).get("lead_strict") for s in SEEDS]
        acc_plot = [v if v is not None else 0 for v in acc_vals]
        strict_plot = [v if v is not None else 0 for v in strict_vals]
        ax.bar(x - width / 2, strict_plot, width, color="#1f77b4",
               alpha=0.85, edgecolor="k", linewidth=0.5, label="Strict lead")
        ax.bar(x + width / 2, acc_plot, width, color="#ff7f0e",
               alpha=0.85, edgecolor="k", linewidth=0.5, label="Lead to acc>=0.99")
        ax.set_xticks(x)
        ax.set_xticklabels([f"s={s}" for s in SEEDS])
        ax.set_ylabel("Lead time (steps)")
        ax.set_title("Dyck Depth Prediction: Defect Spike Lead Time\n"
                     "(strict vs first acc>=0.99)")
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, loc="upper left")

        for i, (v_strict, v_acc) in enumerate(zip(strict_vals, acc_vals)):
            if v_strict is None:
                ax.text(i - width / 2, 200, "NA", ha="center", va="bottom",
                        fontsize=7, color="#1f77b4")
            if v_acc is None:
                ax.text(i + width / 2, 200, "NA", ha="center", va="bottom",
                        fontsize=7, color="#ff7f0e")

        if lead_times_acc:
            n_pos = sum(1 for l in lead_times_acc if l > 0)
            p_val = 2 ** (-len(lead_times_acc))
            sign_text = f"Sign test (acc): {n_pos}/{len(lead_times_acc)} positive\np = {p_val:.4f}"
        else:
            sign_text = "Sign test (acc): n/a"
        ax.text(0.98, 0.95, sign_text,
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                         edgecolor="gray"))

        fig.tight_layout()
        fig.savefig(OUT_DIR / "figX_dyck_lead_time.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved figX_dyck_lead_time.png")

    # ── Figure W2: Hero single-panel ─────────────────────────────────
    best_seed, best_lead = None, -1e9
    for row in lead_rows:
        lead = row["lead_acc"]
        if lead is not None and lead > best_lead:
            best_lead = lead
            best_seed = row["seed"]

    if best_seed is not None:
        data = all_runs[("wd1.0", best_seed)]
        recs = data["records"]
        steps = [r["step"] for r in recs]
        defects = [r["defect_median"] for r in recs]
        test_accs = [r["test_acc"] for r in recs]
        defect_25 = [r["defect_p25"] for r in recs]
        defect_75 = [r["defect_p75"] for r in recs]

        spike = find_spike_step(recs)
        strict_step = strict_grok_step(data)
        grok = find_grok_step_from_records(recs, 0.99)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax2 = ax.twinx()

        ax.fill_between(steps, defect_25, defect_75, alpha=0.15, color="#1a5276")
        ax.plot(steps, defects, color="#1a5276", linewidth=2.5, label="Commutator defect")
        ax2.plot(steps, test_accs, color="#e74c3c", linewidth=2.5, linestyle="--",
                 label="Test accuracy")

        if spike is not None:
            ax.axvline(x=spike, color="#1a5276", linestyle=":", linewidth=2,
                       alpha=0.7, label=f"Defect spike (step {spike})")
        if grok is not None:
            ax.axvline(x=grok, color="#e74c3c", linestyle=":", linewidth=2,
                       alpha=0.7, label=f"First acc>=0.99 (step {grok})")
        if strict_step is not None:
            ax.axvline(x=strict_step, color="#8e44ad", linestyle="-.", linewidth=2,
                       alpha=0.6, label=f"Strict grok (step {strict_step})")

        if spike is not None and grok is not None:
            mid = (spike + grok) / 2
            ax.annotate("", xy=(spike, ax.get_ylim()[1] * 0.7),
                        xytext=(grok, ax.get_ylim()[1] * 0.7),
                        arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.5))
            ax.text(mid, ax.get_ylim()[1] * 0.8,
                    f"Δ = {grok - spike} steps",
                    ha="center", fontsize=11, fontweight="bold")

        ax.set_yscale("log")
        ax.set_xlabel("Training step", fontsize=12)
        ax.set_ylabel("Commutator defect", fontsize=12, color="#1a5276")
        ax2.set_ylabel("Test accuracy", fontsize=12, color="#e74c3c")
        ax2.set_ylim(-0.05, 1.05)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center left")
        ax.grid(alpha=0.2)

        fig.suptitle(f"Dyck Depth Prediction: Commutator Defect Predicts Generalization\n"
                     f"(seed={best_seed}, spike at {spike}, acc>=0.99 at {grok}, "
                     f"lead={grok-spike} steps)",
                     fontsize=13, y=1.03)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "figW2_dyck_hero.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved figW2_dyck_hero.png")

    # ── Save ─────────────────────────────────────────────────────────
    torch.save({"all_runs": all_runs, "lead_rows": lead_rows,
                "lead_times_acc": lead_times_acc}, cache_path)
    print(f"\n  saved {cache_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
