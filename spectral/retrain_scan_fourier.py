#!/usr/bin/env python3
"""
Retrain SCAN models saving full state_dict at key training steps.

Needed for Fourier functional analysis (hidden representation extraction
requires full model state). Uses seed=2024 which groks fastest (~11.5K steps).
"""

import sys, os, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import torch.nn as nn

from scan.grok_sweep import (
    ScanSweepConfig, ScanTransformer,
    download_scan, parse_scan_file, build_vocabs, build_scan_dataset,
    sample_batch, masked_ce_loss, eval_on_dataset, get_device,
    SCAN_DATA_DIR,
)

OUT_DIR = Path(__file__).resolve().parent / "fourier_scan_checkpoints"


def retrain_with_state_dicts(weight_decay: float, seed: int = 2024,
                              steps: int = 15_000, log_every: int = 500):
    """Retrain SCAN and save full state_dict at regular intervals."""
    device = get_device()
    cfg = ScanSweepConfig()
    cfg.SEED = seed
    cfg.WEIGHT_DECAY = weight_decay
    cfg.STEPS = steps

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    download_scan()
    train_cmds, train_acts = parse_scan_file(SCAN_DATA_DIR / "tasks_train_simple.txt")
    test_cmds, test_acts = parse_scan_file(SCAN_DATA_DIR / "tasks_test_simple.txt")

    rng = random.Random(cfg.DATA_SEED)
    indices = list(range(len(train_cmds)))
    rng.shuffle(indices)
    n_train = min(cfg.N_TRAIN, len(train_cmds))
    train_indices = indices[:n_train]
    sub_train_cmds = [train_cmds[i] for i in train_indices]
    sub_train_acts = [train_acts[i] for i in train_indices]

    cmd_vocab, act_vocab = build_vocabs(train_cmds, train_acts, test_cmds, test_acts)

    all_cmds = sub_train_cmds + test_cmds
    all_acts = sub_train_acts + test_acts
    max_cmd_len = max(len(c) for c in all_cmds) + 2
    max_act_len = max(len(a) for a in all_acts) + 2

    train_src, train_tgt_in, train_tgt_out = build_scan_dataset(
        cmd_vocab, act_vocab, sub_train_cmds, sub_train_acts, max_cmd_len, max_act_len)
    test_src, test_tgt_in, test_tgt_out = build_scan_dataset(
        cmd_vocab, act_vocab, test_cmds, test_acts, max_cmd_len, max_act_len)

    n_test_eval = min(cfg.N_TEST_EVAL, len(test_src))
    test_perm = torch.randperm(len(test_src), generator=torch.Generator().manual_seed(cfg.DATA_SEED))
    eval_test_src = test_src[test_perm[:n_test_eval]]
    eval_test_tgt_in = test_tgt_in[test_perm[:n_test_eval]]
    eval_test_tgt_out = test_tgt_out[test_perm[:n_test_eval]]

    print(f"  Vocab: {cmd_vocab.size} commands, {act_vocab.size} actions")
    print(f"  Dataset: {len(train_src)} train, {n_test_eval} test-eval")
    print(f"  Max lengths: cmd={max_cmd_len}, act={max_act_len}")

    model = ScanTransformer(
        src_vocab_size=cmd_vocab.size, tgt_vocab_size=act_vocab.size,
        max_src_len=max_cmd_len, max_tgt_len=max_act_len,
        d_model=cfg.D_MODEL, n_layers=cfg.N_LAYERS,
        n_heads=cfg.N_HEADS, d_ff=cfg.D_FF, dropout=cfg.DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params, device={device}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2),
    )

    pad_id = 0
    snapshots = []
    metrics = []

    snapshots.append({"step": 0, "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()}})

    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        src, tgt_in, tgt_out = sample_batch(train_src, train_tgt_in, train_tgt_out, cfg.BATCH_SIZE, device)
        src_pad_mask = (src == pad_id)
        tgt_pad_mask = (tgt_in == pad_id)
        logits = model(src, tgt_in, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
        loss = masked_ce_loss(logits, tgt_out)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        if step % log_every == 0:
            train_loss, train_acc, train_seq = eval_on_dataset(
                model, train_src, train_tgt_in, train_tgt_out, device, pad_id)
            test_loss, test_acc, test_seq = eval_on_dataset(
                model, eval_test_src, eval_test_tgt_in, eval_test_tgt_out, device, pad_id)
            metrics.append({
                "step": step, "train_loss": train_loss, "train_acc": train_acc,
                "train_seq_acc": train_seq, "test_loss": test_loss,
                "test_acc": test_acc, "test_seq_acc": test_seq,
            })
            snapshots.append({
                "step": step,
                "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            })
            grok_marker = " *** GROKKED ***" if test_seq >= 0.95 else ""
            print(f"  step {step:5d}  train_seq={train_seq:.3f}  test_seq={test_seq:.3f}"
                  f"  [{time.time()-t0:.1f}s]{grok_marker}")

    return {
        "cfg": asdict(cfg),
        "snapshots": snapshots,
        "metrics": metrics,
        "cmd_vocab": cmd_vocab,
        "act_vocab": act_vocab,
        "max_cmd_len": max_cmd_len,
        "max_act_len": max_act_len,
        "train_src": train_src, "train_tgt_in": train_tgt_in, "train_tgt_out": train_tgt_out,
        "test_src": eval_test_src, "test_tgt_in": eval_test_tgt_in, "test_tgt_out": eval_test_tgt_out,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for wd in [1.0, 0.0]:
        tag = "grok" if wd == 1.0 else "memo"
        print(f"\n{'='*60}")
        print(f"Retraining SCAN wd={wd} ({tag}) for Fourier analysis")
        print(f"{'='*60}")

        result = retrain_with_state_dicts(weight_decay=wd)
        out_path = OUT_DIR / f"scan_{tag}_fourier.pt"
        torch.save(result, out_path)
        print(f"\nSaved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
        print(f"  {len(result['snapshots'])} snapshots, {len(result['metrics'])} metric records")
        if result['metrics']:
            final = result['metrics'][-1]
            print(f"  Final: train_seq={final['train_seq_acc']:.3f}, test_seq={final['test_seq_acc']:.3f}")


if __name__ == "__main__":
    main()
