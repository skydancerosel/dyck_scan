#!/usr/bin/env python3
"""
Retrain Dyck models saving full state_dict at key training steps.

Needed for Fourier functional analysis (hidden representation extraction
requires full model, not just attention weight snapshots).

Saves: grokked (wd=1.0) and memorized (wd=0.0) models with state_dicts
at every 100 steps up to 10K steps.
"""

import sys, os, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import torch.nn as nn

from dyck.grok_sweep import (
    DyckSweepConfig, DyckTransformerLM, VOCAB_SIZE,
    build_depth_dataset, split_dataset, sample_batch,
    masked_ce_loss, eval_on_dataset, get_device,
)

OUT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"


def retrain_with_state_dicts(weight_decay: float, seed: int = 42,
                              steps: int = 10_000, log_every: int = 100):
    """Retrain and save full state_dict at regular intervals."""
    device = get_device()
    cfg = DyckSweepConfig()
    cfg.SEED = seed
    cfg.WEIGHT_DECAY = weight_decay
    cfg.STEPS = steps

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    # Build dataset (same DATA_SEED=0 as original)
    X_all, Y_all = build_depth_dataset(
        n_seqs=cfg.N_TOTAL, max_pairs=cfg.MAX_PAIRS,
        ctx_len=cfg.CTX_LEN, seed=cfg.DATA_SEED
    )
    frac = cfg.N_TRAIN / cfg.N_TOTAL
    train_x, train_y, test_x, test_y = split_dataset(
        X_all, Y_all, frac_train=frac, seed=cfg.DATA_SEED
    )

    model = DyckTransformerLM(
        vocab_size=VOCAB_SIZE,
        ctx_len=max(cfg.CTX_LEN, cfg.CTX_LEN_OOD),
        d_model=cfg.D_MODEL, n_layers=cfg.N_LAYERS,
        n_heads=cfg.N_HEADS, d_ff=cfg.D_FF,
        dropout=cfg.DROPOUT, n_classes=cfg.N_CLASSES,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2),
    )

    snapshots = []
    metrics = []

    # Save initial state
    snapshots.append({"step": 0, "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()}})

    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        bx, by = sample_batch(train_x, train_y, cfg.BATCH_SIZE, device)
        logits = model(bx)
        loss = masked_ce_loss(logits, by)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        if step % log_every == 0:
            train_loss, train_acc = eval_on_dataset(model, train_x, train_y, device)
            test_loss, test_acc = eval_on_dataset(model, test_x, test_y, device)
            metrics.append({
                "step": step, "train_loss": train_loss, "train_acc": train_acc,
                "test_loss": test_loss, "test_acc": test_acc,
            })
            snapshots.append({
                "step": step,
                "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            })
            grok_marker = " *** GROKKED ***" if test_acc >= 0.95 else ""
            if step % 500 == 0 or test_acc >= 0.95:
                print(f"  step {step:5d}  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}"
                      f"  [{time.time()-t0:.1f}s]{grok_marker}")

    return {
        "cfg": asdict(cfg),
        "snapshots": snapshots,
        "metrics": metrics,
        "train_x": train_x, "train_y": train_y,
        "test_x": test_x, "test_y": test_y,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for wd in [1.0, 0.0]:
        tag = "grok" if wd == 1.0 else "memo"
        print(f"\n{'='*60}")
        print(f"Retraining Dyck wd={wd} ({tag}) for Fourier analysis")
        print(f"{'='*60}")

        result = retrain_with_state_dicts(weight_decay=wd)
        out_path = OUT_DIR / f"dyck_{tag}_fourier.pt"
        torch.save(result, out_path)
        print(f"\nSaved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
        print(f"  {len(result['snapshots'])} snapshots, {len(result['metrics'])} metric records")
        final = result['metrics'][-1]
        print(f"  Final: train_acc={final['train_acc']:.3f}, test_acc={final['test_acc']:.3f}")


if __name__ == "__main__":
    main()
