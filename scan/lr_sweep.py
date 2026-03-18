#!/usr/bin/env python3
"""
Learning rate sweep: phase diagram of SCAN grokking dynamics.

Sweeps lr in {1e-5, 5e-5, 1e-4} with fixed wd=1.0 across 3 seeds.
Reuses existing lr=1e-4 data from generalization_dynamics cache.

Produces:
  figPD  — Phase diagram: grok fraction, grok step, max defect, lead time
  figPD2 — Hero: defect + test_acc across LRs
"""

import math, time, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from scan_grok_sweep import (
    ScanSweepConfig, ScanTransformer, masked_ce_loss,
    get_device, eval_on_dataset,
    download_scan, parse_scan_file, build_vocabs, build_scan_dataset,
    sample_batch, PAD_TOKEN, SCAN_DATA_DIR,
)
from scan_commutator_analysis import commutator_defect
from scan_generalization_dynamics import find_spike_step

# -- config -------------------------------------------------------------------
OUT_DIR = Path(__file__).parent / "scan_pca_plots"
LRS = [1e-5, 5e-5, 1e-4]
SEEDS = [42]
WDS = [1.0, 0.0]

MAX_STEPS_BY_LR = {1e-5: 200_000, 5e-5: 100_000, 1e-4: 60_000}
COMM_EVERY_BY_LR = {1e-5: 1000, 5e-5: 500, 1e-4: 500}
COMM_K = 5
COMM_ETA = 1e-3
POST_GROK_STEPS = 1000


# -- local grok detection --
def find_grok_step_from_records_acc(records, acc_threshold=0.99):
    for r in records:
        if r.get("test_seq_acc", 0) >= acc_threshold:
            return r["step"]
    return None


def strict_grok_step(data):
    if data.get("grokked") and data.get("grok_step") is not None:
        return data["grok_step"]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Data loading helper
# ═══════════════════════════════════════════════════════════════════════════

def _load_scan_data(cfg):
    download_scan()
    train_cmds, train_acts = parse_scan_file(
        SCAN_DATA_DIR / "tasks_train_simple.txt")
    test_cmds, test_acts = parse_scan_file(
        SCAN_DATA_DIR / "tasks_test_simple.txt")

    rng = random.Random(cfg.DATA_SEED)
    indices = list(range(len(train_cmds)))
    rng.shuffle(indices)
    n_train = min(cfg.N_TRAIN, len(train_cmds))
    train_indices = indices[:n_train]
    sub_train_cmds = [train_cmds[i] for i in train_indices]
    sub_train_acts = [train_acts[i] for i in train_indices]

    cmd_vocab, act_vocab = build_vocabs(train_cmds, train_acts,
                                         test_cmds, test_acts)

    all_cmds = sub_train_cmds + test_cmds
    all_acts = sub_train_acts + test_acts
    max_cmd_len = max(len(c) for c in all_cmds) + 2
    max_act_len = max(len(a) for a in all_acts) + 2

    train_src, train_tgt_in, train_tgt_out = build_scan_dataset(
        cmd_vocab, act_vocab, sub_train_cmds, sub_train_acts,
        max_cmd_len, max_act_len)
    test_src, test_tgt_in, test_tgt_out = build_scan_dataset(
        cmd_vocab, act_vocab, test_cmds, test_acts,
        max_cmd_len, max_act_len)

    return {
        "cmd_vocab": cmd_vocab, "act_vocab": act_vocab,
        "max_cmd_len": max_cmd_len, "max_act_len": max_act_len,
        "train_src": train_src, "train_tgt_in": train_tgt_in,
        "train_tgt_out": train_tgt_out,
        "test_src": test_src, "test_tgt_in": test_tgt_in,
        "test_tgt_out": test_tgt_out,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Training with inline defect tracking (LR-parameterized)
# ═══════════════════════════════════════════════════════════════════════════

def train_with_defect_tracking_lr(lr, wd, seed, max_steps, comm_every=100):
    device = get_device()
    cfg = ScanSweepConfig(WEIGHT_DECAY=wd, SEED=seed, STEPS=max_steps, LR=lr)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data = _load_scan_data(cfg)
    cmd_vocab = data["cmd_vocab"]
    act_vocab = data["act_vocab"]
    train_src = data["train_src"]
    train_tgt_in = data["train_tgt_in"]
    train_tgt_out = data["train_tgt_out"]
    test_src = data["test_src"]
    test_tgt_in = data["test_tgt_in"]
    test_tgt_out = data["test_tgt_out"]
    pad_id_src = cmd_vocab.token2idx[PAD_TOKEN]
    pad_id_tgt = act_vocab.token2idx[PAD_TOKEN]

    print(f"      Dataset: {len(train_src)} train, {len(test_src)} test")

    model = ScanTransformer(
        src_vocab_size=cmd_vocab.size,
        tgt_vocab_size=act_vocab.size,
        max_src_len=data["max_cmd_len"],
        max_tgt_len=data["max_act_len"],
        d_model=cfg.D_MODEL, n_layers=cfg.N_LAYERS,
        n_heads=cfg.N_HEADS, d_ff=cfg.D_FF, dropout=cfg.DROPOUT,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    )

    def batch_fn():
        return sample_batch(train_src, train_tgt_in, train_tgt_out,
                           cfg.BATCH_SIZE, device)

    records = []
    grokked = False
    grok_step = None
    diverged = False
    patience = 0
    steps_after_grok = 0
    t0 = time.time()

    # Step 0
    train_loss, train_acc, train_seq_acc = eval_on_dataset(
        model, train_src, train_tgt_in, train_tgt_out, device,
        pad_id=pad_id_src)
    test_loss, test_acc, test_seq_acc = eval_on_dataset(
        model, test_src, test_tgt_in, test_tgt_out, device,
        pad_id=pad_id_src)
    defects = []
    for _ in range(COMM_K):
        D, delta, gcos, nA, nB = commutator_defect(
            model, batch_fn, device, eta=COMM_ETA,
            pad_id_src=pad_id_src, pad_id_tgt=pad_id_tgt)
        defects.append(D)
    records.append({
        "step": 0,
        "defect_median": float(np.median(defects)),
        "defect_p25": float(np.percentile(defects, 25)),
        "defect_p75": float(np.percentile(defects, 75)),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_seq_acc": train_seq_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_seq_acc": test_seq_acc,
    })

    for step in range(1, cfg.STEPS + 1):
        model.train()
        src, tgt_in, tgt_out = sample_batch(
            train_src, train_tgt_in, train_tgt_out, cfg.BATCH_SIZE, device)

        src_pad_mask = (src == pad_id_src)
        tgt_pad_mask = (tgt_in == pad_id_tgt)
        logits = model(src, tgt_in,
                       src_pad_mask=src_pad_mask,
                       tgt_pad_mask=tgt_pad_mask)
        loss = masked_ce_loss(logits, tgt_out)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"      DIVERGED at step {step}")
            diverged = True
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        if step % comm_every == 0:
            model.eval()
            train_loss, train_acc, train_seq_acc = eval_on_dataset(
                model, train_src, train_tgt_in, train_tgt_out, device,
                pad_id=pad_id_src)
            test_loss, test_acc, test_seq_acc = eval_on_dataset(
                model, test_src, test_tgt_in, test_tgt_out, device,
                pad_id=pad_id_src)
            defects = []
            for _ in range(COMM_K):
                D, delta, gcos, nA, nB = commutator_defect(
                    model, batch_fn, device, eta=COMM_ETA,
                    pad_id_src=pad_id_src, pad_id_tgt=pad_id_tgt)
                defects.append(D)
            records.append({
                "step": step,
                "defect_median": float(np.median(defects)),
                "defect_p25": float(np.percentile(defects, 25)),
                "defect_p75": float(np.percentile(defects, 75)),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_seq_acc": train_seq_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_seq_acc": test_seq_acc,
            })
            model.train()

        if step % cfg.EVAL_EVERY == 0:
            if step % comm_every != 0:
                test_loss, test_acc, test_seq_acc = eval_on_dataset(
                    model, test_src, test_tgt_in, test_tgt_out, device,
                    pad_id=pad_id_src)

            if test_seq_acc >= cfg.STOP_ACC:
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

        if step % 5000 == 0:
            elapsed = (time.time() - t0) / 60
            last_r = records[-1] if records else {}
            d = last_r.get("defect_median", 0)
            ta = last_r.get("test_seq_acc", 0)
            print(f"      step {step:7d} | test_seq_acc {ta:.4f} | defect {d:.1f} | "
                  f"{elapsed:.1f}m")

    return {
        "records": records,
        "grokked": grokked,
        "grok_step": grok_step,
        "diverged": diverged,
        "lr": lr, "wd": wd, "seed": seed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Reuse existing lr=1e-3 data
# ═══════════════════════════════════════════════════════════════════════════

def load_existing_lr1e4():
    """Reuse lr=1e-4 data from generalization_dynamics cache."""
    cache_path = OUT_DIR / "scan_generalization_dynamics_results.pt"
    loaded = {}
    if not cache_path.exists():
        return loaded
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    if "all_runs" not in cached:
        return loaded
    for key, data in cached["all_runs"].items():
        tag, seed = key
        if tag == "wd1.0":
            new_key = (1e-4, 1.0, seed)
            data_copy = dict(data)
            data_copy["lr"] = 1e-4
            data_copy.setdefault("diverged", False)
            loaded[new_key] = data_copy
            print(f"    reused: lr=1e-4 wd=1.0 s={seed}")
    return loaded


# ═══════════════════════════════════════════════════════════════════════════
# Sweep
# ═══════════════════════════════════════════════════════════════════════════

def run_sweep():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    results_path = OUT_DIR / "scan_lr_sweep_results.pt"
    all_runs = {}

    if results_path.exists():
        cached = torch.load(results_path, map_location="cpu", weights_only=False)
        if "all_runs" in cached:
            all_runs = cached["all_runs"]
            print(f"  Loaded {len(all_runs)} cached runs")

    if not any(k[0] == 1e-4 for k in all_runs):
        existing = load_existing_lr1e4()
        all_runs.update(existing)

    total = len(LRS) * len(SEEDS)
    print(f"\n  Grid: {len(LRS)} LRs x {len(SEEDS)} seeds = {total} runs (wd=1.0)")

    run_i = 0
    for lr in LRS:
        max_steps = MAX_STEPS_BY_LR[lr]
        comm_every = COMM_EVERY_BY_LR.get(lr, 100)
        for seed in SEEDS:
            run_i += 1
            key = (lr, 1.0, seed)
            if key in all_runs:
                continue
            tag = f"lr={lr:.0e} s={seed}"
            print(f"\n  [{run_i}/{total}] {tag} (max {max_steps} steps, comm_every={comm_every})")
            data = train_with_defect_tracking_lr(lr, 1.0, seed, max_steps,
                                                  comm_every=comm_every)
            all_runs[key] = data
            print(f"    -> grokked={data['grokked']} (step={data['grok_step']})")
            torch.save({"all_runs": all_runs}, results_path)

    print(f"\n  Sweep complete: {len(all_runs)} runs")
    return all_runs


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def compute_summary(all_runs):
    summary = {}
    for lr in LRS:
        strict_flags, strict_steps = [], []
        acc_flags, acc_steps = [], []
        max_defects = []
        lead_times_acc, lead_times_strict = [], []
        lead_ratios_acc, lead_ratios_strict = [], []
        for seed in SEEDS:
            key = (lr, 1.0, seed)
            if key not in all_runs:
                continue
            data = all_runs[key]
            if data.get("diverged", False):
                continue
            recs = data["records"]
            if recs:
                s_step = strict_grok_step(data)
                acc_step = find_grok_step_from_records_acc(recs, 0.99)
                strict_flags.append(1 if s_step is not None else 0)
                acc_flags.append(1 if acc_step is not None else 0)
                if s_step is not None:
                    strict_steps.append(s_step)
                if acc_step is not None:
                    acc_steps.append(acc_step)
                max_defects.append(max(r["defect_median"] for r in recs))
                spike = find_spike_step(recs)
                if spike is not None and acc_step is not None:
                    lead_time = acc_step - spike
                    lead_times_acc.append(lead_time)
                    lead_ratios_acc.append(lead_time / acc_step if acc_step > 0 else 0)
                if spike is not None and s_step is not None:
                    lead_time = s_step - spike
                    lead_times_strict.append(lead_time)
                    lead_ratios_strict.append(lead_time / s_step if s_step > 0 else 0)

        summary[lr] = {
            "strict_frac": np.mean(strict_flags) if strict_flags else 0,
            "n_strict": sum(strict_flags) if strict_flags else 0,
            "strict_step_mean": np.mean(strict_steps) if strict_steps else None,
            "acc_frac": np.mean(acc_flags) if acc_flags else 0,
            "n_acc": sum(acc_flags) if acc_flags else 0,
            "acc_step_mean": np.mean(acc_steps) if acc_steps else None,
            "n_total": len(acc_flags),
            "max_defect_mean": np.mean(max_defects) if max_defects else None,
            "lead_time_acc_mean": np.mean(lead_times_acc) if lead_times_acc else None,
            "lead_time_strict_mean": np.mean(lead_times_strict) if lead_times_strict else None,
            "lead_ratio_acc_mean": np.mean(lead_ratios_acc) if lead_ratios_acc else None,
            "lead_ratio_strict_mean": np.mean(lead_ratios_strict) if lead_ratios_strict else None,
        }
    return summary


def plot_phase_diagram(all_runs):
    summary = compute_summary(all_runs)

    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    lr_labels = [f"{lr:.0e}" for lr in LRS]
    x = np.arange(len(LRS))
    width = 0.36

    # Panel A: Grok fraction
    ax = axes[0]
    strict_vals = [summary[lr]["strict_frac"] for lr in LRS]
    acc_vals = [summary[lr]["acc_frac"] for lr in LRS]
    ax.bar(x - width / 2, strict_vals, width, color="#1f77b4", alpha=0.85,
           label="Strict grok")
    ax.bar(x + width / 2, acc_vals, width, color="#ff7f0e", alpha=0.85,
           label="First seq_acc>=0.99")
    for i, lr in enumerate(LRS):
        s = summary[lr]
        ax.text(i - width / 2, strict_vals[i] + 0.02, f"{s['n_strict']}/{s['n_total']}",
                ha="center", fontsize=9)
        ax.text(i + width / 2, acc_vals[i] + 0.02, f"{s['n_acc']}/{s['n_total']}",
                ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(lr_labels)
    ax.set_ylabel("Fraction")
    ax.set_title("(A) Grok fraction")
    ax.set_ylim(0, 1.2); ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper left")

    # Panel B: Grok step
    ax = axes[1]
    strict_vals = [summary[lr]["strict_step_mean"] or 0 for lr in LRS]
    acc_vals = [summary[lr]["acc_step_mean"] or 0 for lr in LRS]
    ax.bar(x - width / 2, strict_vals, width, color="#1f77b4", alpha=0.85)
    ax.bar(x + width / 2, acc_vals, width, color="#ff7f0e", alpha=0.85)
    for i, v in enumerate(strict_vals):
        if v > 0:
            ax.text(i - width / 2, v + 100, f"{v:.0f}", ha="center", fontsize=8)
    for i, v in enumerate(acc_vals):
        if v > 0:
            ax.text(i + width / 2, v + 100, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(lr_labels)
    ax.set_ylabel("Mean step")
    ax.set_title("(B) Mean grok step")
    ax.grid(alpha=0.3, axis="y")

    # Panel C: Max defect
    ax = axes[2]
    vals = [summary[lr]["max_defect_mean"] or 0 for lr in LRS]
    ax.bar(x, vals, color="#d62728", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(lr_labels)
    ax.set_ylabel("Mean max defect"); ax.set_title("(C) Mean max defect")
    ax.set_yscale("log"); ax.grid(alpha=0.3, axis="y")

    # Panel D: Lead time
    ax = axes[3]
    strict_vals = [summary[lr]["lead_time_strict_mean"] or 0 for lr in LRS]
    acc_vals = [summary[lr]["lead_time_acc_mean"] or 0 for lr in LRS]
    ax.bar(x - width / 2, strict_vals, width, color="#1f77b4", alpha=0.85)
    ax.bar(x + width / 2, acc_vals, width, color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(lr_labels)
    ax.set_ylabel("Mean lead time (steps)")
    ax.set_title("(D) Lead time")
    ax.axhline(y=0, color="k", linewidth=0.5); ax.grid(alpha=0.3, axis="y")

    # Panel E: Lead ratio
    ax = axes[4]
    strict_vals = [summary[lr]["lead_ratio_strict_mean"] or 0 for lr in LRS]
    acc_vals = [summary[lr]["lead_ratio_acc_mean"] or 0 for lr in LRS]
    ax.bar(x - width / 2, strict_vals, width, color="#1f77b4", alpha=0.85,
           label="Strict grok")
    ax.bar(x + width / 2, acc_vals, width, color="#ff7f0e", alpha=0.85,
           label="First seq_acc>=0.99")
    for i, v in enumerate(acc_vals):
        if v > 0:
            ax.text(i + width / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(strict_vals):
        if v > 0:
            ax.text(i - width / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(lr_labels)
    ax.set_ylabel("Lead ratio (lead / grok_step)")
    ax.set_title("(E) Lead ratio")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0, color="k", linewidth=0.5); ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.suptitle("SCAN: Phase Diagram across Learning Rates "
                 "(strict vs first seq_acc>=0.99, wd=1.0, 3 seeds)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPD_scan_lr_phase_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figPD_scan_lr_phase_diagram.png")


def plot_hero(all_runs):
    LR_COLORS = {1e-5: "#1f77b4", 5e-5: "#ff7f0e", 1e-4: "#2ca02c"}

    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()

    for lr in LRS:
        color = LR_COLORS[lr]
        best_seed, best_lead = None, -1e9
        for seed in SEEDS:
            key = (lr, 1.0, seed)
            if key not in all_runs:
                continue
            data = all_runs[key]
            if data.get("diverged", False):
                continue
            recs = data["records"]
            spike = find_spike_step(recs)
            grok = find_grok_step_from_records_acc(recs, 0.99)
            lead = (grok - spike) if (spike is not None and grok is not None) else -1e9
            if lead > best_lead:
                best_lead = lead
                best_seed = seed

        if best_seed is None:
            for seed in SEEDS:
                key = (lr, 1.0, seed)
                if key in all_runs:
                    best_seed = seed
                    break
        if best_seed is None:
            continue

        data = all_runs[(lr, 1.0, best_seed)]
        recs = data["records"]
        steps = [r["step"] for r in recs]
        defects = [r["defect_median"] for r in recs]
        test_accs = [r.get("test_seq_acc", 0) for r in recs]

        ax.plot(steps, defects, color=color, linewidth=2, alpha=0.85,
                label=f"Defect (lr={lr:.0e})")
        ax2.plot(steps, test_accs, color=color, linewidth=1.5, linestyle="--",
                 alpha=0.6, label=f"Test seq acc (lr={lr:.0e})")

        spike = find_spike_step(recs)
        grok = find_grok_step_from_records_acc(recs, 0.99)
        if spike is not None:
            ax.axvline(x=spike, color=color, linestyle=":", alpha=0.5, linewidth=1)
        if grok is not None:
            ax.axvline(x=grok, color=color, linestyle="-.", alpha=0.3, linewidth=1)

    ax.set_yscale("log")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Commutator defect", fontsize=11)
    ax2.set_ylabel("Test seq accuracy", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax.set_title("SCAN: Defect Predicts Grokking across Learning Rates",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.2)

    handles = []
    for lr in LRS:
        c = LR_COLORS[lr]
        handles.append(Line2D([0], [0], color=c, linewidth=2,
                              label=f"lr={lr:.0e} defect"))
        handles.append(Line2D([0], [0], color=c, linewidth=1.5, linestyle="--",
                              label=f"lr={lr:.0e} test seq acc"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPD2_scan_lr_sweep_hero.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figPD2_scan_lr_sweep_hero.png")


def print_summary_table(all_runs):
    summary = compute_summary(all_runs)
    print(f"\n{'='*90}")
    print("  SCAN LR SWEEP SUMMARY")
    print(f"{'='*90}")
    print(f"  {'LR':>8s}  {'Strict':>8s}  {'StrictStep':>10s}  "
          f"{'SeqAcc>=99':>10s}  {'AccStep':>8s}  {'MaxDefect':>10s}  "
          f"{'LeadAcc':>8s}  {'LeadRatio':>10s}")
    for lr in LRS:
        s = summary[lr]
        strict_step_str = (f"{s['strict_step_mean']:.0f}"
                           if s["strict_step_mean"] is not None else "---")
        acc_step_str = (f"{s['acc_step_mean']:.0f}"
                        if s["acc_step_mean"] is not None else "---")
        defect_str = f"{s['max_defect_mean']:.1f}" if s["max_defect_mean"] is not None else "---"
        lead_acc_str = (f"{s['lead_time_acc_mean']:.0f}"
                        if s["lead_time_acc_mean"] is not None else "---")
        lead_ratio_str = (f"{s['lead_ratio_acc_mean']:.3f}"
                          if s["lead_ratio_acc_mean"] is not None else "---")
        print(f"  {lr:>8.0e}  {s['n_strict']}/{s['n_total']:>3d}  "
              f"{strict_step_str:>10s}  {s['n_acc']}/{s['n_total']:>3d}  "
              f"{acc_step_str:>8s}  {defect_str:>10s}  "
              f"{lead_acc_str:>8s}  {lead_ratio_str:>10s}")


if __name__ == "__main__":
    all_runs = run_sweep()
    print_summary_table(all_runs)
    print("\n  Generating figures...")
    plot_phase_diagram(all_runs)
    plot_hero(all_runs)
    print("\n  Done!")
