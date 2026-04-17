#!/usr/bin/env python3
"""
Retrain Dyck s=42 WD=1.0 with dense checkpointing (every 50 steps)
for better Gram matrix resolution around the grokking transition.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from dataclasses import asdict
from dyck.grok_sweep import (
    DyckSweepConfig, run_single
)
import torch

OUT_DIR = Path(__file__).resolve().parent.parent / "dyck_sweep_results"


def main():
    cfg = DyckSweepConfig()
    cfg.SEED = 42
    cfg.WEIGHT_DECAY = 1.0
    cfg.MODEL_LOG_EVERY = 50   # 4x denser than default 200

    print(f"Retraining Dyck s=42 WD=1.0 with MODEL_LOG_EVERY={cfg.MODEL_LOG_EVERY}")
    result = run_single(cfg)

    out_path = OUT_DIR / "dyck_wd1.0_s42_dense.pt"
    torch.save(result, out_path)
    print(f"\nSaved: {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
    print(f"  {len(result['attn_logs'])} checkpoints, grokked={result['grokked']}")


if __name__ == "__main__":
    main()
