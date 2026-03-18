#!/usr/bin/env python3
"""
Dyck-language grokking sweep for PCA eigenanalysis.

Replicates the experimental pipeline of the integrability-grokking paper
(Xu, 2026) but on the Dyck-1 (balanced parentheses) depth-prediction task.

Task:  Given a prefix of parentheses, predict the stack depth at each
       position (causal / autoregressive).  This is a deterministic map
       from input to output, analogous to modular arithmetic.

"Grokking" setup:
  - Pre-generate a FIXED training set of balanced-parentheses sequences.
  - Small training split (~50 seqs) vs large test split (~5000 seqs).
  - The model first memorises the training set (100% train acc) while test
    accuracy stays lower; later, generalisation kicks in (grokking).
  - Weight decay is the critical control variable: WD>0 → grokking,
    WD=0 → memorisation only (test acc stalls or degrades).

Sweep dimensions:
  - Weight decay: {1.0, 0.0}
  - 3 random seeds

All results saved to dyck_sweep_results/ as individual .pt files.
"""

import math, time, random, json, sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Dyck-1 data generation
# ═══════════════════════════════════════════════════════════════════════════

TOK_OPEN = 0    # "("
TOK_CLOSE = 1   # ")"
TOK_PAD = 2     # padding
VOCAB_SIZE = 3


def gen_balanced_parentheses(max_pairs: int, rng: random.Random) -> List[int]:
    """Generate a balanced parentheses string with <= max_pairs pairs."""
    n_pairs = rng.randint(1, max_pairs)
    seq = []
    open_used = 0
    stack = 0
    for _ in range(2 * n_pairs):
        can_open = open_used < n_pairs
        can_close = stack > 0
        if can_open and can_close:
            if rng.random() < 0.55:
                seq.append(TOK_OPEN); open_used += 1; stack += 1
            else:
                seq.append(TOK_CLOSE); stack -= 1
        elif can_open:
            seq.append(TOK_OPEN); open_used += 1; stack += 1
        else:
            seq.append(TOK_CLOSE); stack -= 1
    assert stack == 0
    return seq


def build_depth_dataset(n_seqs: int, max_pairs: int, ctx_len: int, seed: int):
    """
    Build a fixed dataset of (sequence, depth-labels) pairs.
    Returns (X, Y) where X is [N, ctx_len] input tokens and
    Y is [N, ctx_len] depth labels (-100 for padding = ignore in CE loss).
    """
    rng = random.Random(seed)
    X = torch.full((n_seqs, ctx_len), TOK_PAD, dtype=torch.long)
    Y = torch.full((n_seqs, ctx_len), -100, dtype=torch.long)
    for i in range(n_seqs):
        seq = gen_balanced_parentheses(max_pairs, rng)[:ctx_len]
        stack = 0
        for j, tok in enumerate(seq):
            X[i, j] = tok
            if tok == TOK_OPEN:
                stack += 1
            else:
                stack -= 1
            Y[i, j] = stack
    return X, Y


def split_dataset(X, Y, frac_train: float, seed: int):
    """Shuffle and split into train/test."""
    n = len(X)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    X, Y = X[perm], Y[perm]
    n_train = int(frac_train * n)
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def sample_batch(data_x, data_y, batch_size, device):
    """Sample a random batch from a fixed dataset."""
    idx = torch.randint(0, len(data_x), (batch_size,))
    return data_x[idx].to(device), data_y[idx].to(device)


# Kept for backward compatibility with analysis scripts
def make_batch(batch_size, ctx_len, max_pairs, device, rng):
    """Generate fresh random batch (for OOD eval)."""
    X = torch.full((batch_size, ctx_len), TOK_PAD, dtype=torch.long)
    Y = torch.full((batch_size, ctx_len), -100, dtype=torch.long)
    for i in range(batch_size):
        seq = gen_balanced_parentheses(max_pairs, rng)[:ctx_len]
        stack = 0
        for j, tok in enumerate(seq):
            X[i, j] = tok
            if tok == TOK_OPEN:
                stack += 1
            else:
                stack -= 1
            Y[i, j] = stack
    return X.to(device), Y.to(device)


def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy ignoring -100 targets (padding)."""
    B, T, V = logits.shape
    logits_flat = logits.view(B * T, V)
    targets_flat = targets.view(B * T)
    mask = targets_flat != -100
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.cross_entropy(logits_flat[mask], targets_flat[mask])


def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-token accuracy ignoring padding (-100)."""
    B, T, V = logits.shape
    preds = logits.argmax(dim=-1)  # [B, T]
    mask = targets != -100
    if mask.sum() == 0:
        return 0.0
    correct = ((preds == targets) & mask).sum().item()
    return correct / mask.sum().item()


# ═══════════════════════════════════════════════════════════════════════════
# Transformer model — depth predictor (causal)
# ═══════════════════════════════════════════════════════════════════════════

class DyckTransformerLM(nn.Module):
    """
    Causal Transformer for Dyck-1 depth prediction.
    Uses nn.TransformerEncoder with causal mask.
    Output: depth class at each position (0..max_depth).
    """
    def __init__(self, vocab_size: int, ctx_len: int, d_model: int,
                 n_layers: int, n_heads: int, d_ff: int, dropout: float,
                 n_classes: int = 13):
        super().__init__()
        self.ctx_len = ctx_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(ctx_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        h = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        h = self.ln(h)
        return self.head(h)  # [B, T, n_classes]


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DyckSweepConfig:
    # Model
    D_MODEL: int = 128
    N_LAYERS: int = 2
    N_HEADS: int = 4
    D_FF: int = 256
    DROPOUT: float = 0.0
    N_CLASSES: int = 13   # depth 0..12

    # Training
    LR: float = 1e-3
    BATCH_SIZE: int = 50   # = full training set
    STEPS: int = 20_000
    EVAL_EVERY: int = 100
    MODEL_LOG_EVERY: int = 200
    GRAD_CLIP: float = 1.0
    ADAM_BETA1: float = 0.9
    ADAM_BETA2: float = 0.98

    # Data — fixed dataset for grokking
    N_TOTAL: int = 5050     # total generated sequences
    N_TRAIN: int = 50       # very small → forces memorisation
    CTX_LEN: int = 24       # context length
    MAX_PAIRS: int = 12     # max nesting depth
    DATA_SEED: int = 0      # fixed across all runs (same dataset)

    # OOD eval — longer sequences (fresh random, not from dataset)
    CTX_LEN_OOD: int = 48
    MAX_PAIRS_OOD: int = 20

    # Stopping
    STOP_ACC: float = 0.99  # stop if test_acc >= this
    STOP_PATIENCE: int = 10

    # Sweep params (set per run)
    WEIGHT_DECAY: float = 1.0
    SEED: int = 42

    # Eval
    EVAL_BS_OOD: int = 256
    EVAL_BATCHES_OOD: int = 10


OUT_DIR = Path(__file__).parent / "dyck_sweep_results"


# ═══════════════════════════════════════════════════════════════════════════
# Device
# ═══════════════════════════════════════════════════════════════════════════

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_on_dataset(model, data_x, data_y, device, batch_size=512):
    """Evaluate loss and accuracy on a full fixed dataset."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n = len(data_x)
    for i in range(0, n, batch_size):
        bx = data_x[i:i+batch_size].to(device)
        by = data_y[i:i+batch_size].to(device)
        logits = model(bx)
        loss = masked_ce_loss(logits, by)
        mask = by != -100
        preds = logits.argmax(dim=-1)
        total_correct += ((preds == by) & mask).sum().item()
        total_tokens += mask.sum().item()
        total_loss += loss.item() * mask.sum().item()
    avg_loss = total_loss / max(total_tokens, 1)
    avg_acc = total_correct / max(total_tokens, 1)
    model.train()
    return avg_loss, avg_acc


@torch.no_grad()
def eval_loss_acc(model, n_batches, batch_size, ctx_len, max_pairs, device, rng):
    """Evaluate on fresh random batches (for OOD eval)."""
    model.eval()
    losses, accs = [], []
    for _ in range(n_batches):
        x, y = make_batch(batch_size, ctx_len, max_pairs, device, rng)
        logits = model(x)
        losses.append(masked_ce_loss(logits, y).item())
        accs.append(masked_accuracy(logits, y))
    model.train()
    return sum(losses) / len(losses), sum(accs) / len(accs)


@torch.no_grad()
def extract_attn_matrices(model):
    """Extract attention + MLP weight matrices from all layers."""
    logs = []
    for i, layer in enumerate(model.encoder.layers):
        attn = layer.self_attn
        d = attn.embed_dim
        if attn._qkv_same_embed_dim:
            Wq = attn.in_proj_weight[:d]
            Wk = attn.in_proj_weight[d:2*d]
            Wv = attn.in_proj_weight[2*d:]
        else:
            Wq = attn.q_proj_weight
            Wk = attn.k_proj_weight
            Wv = attn.v_proj_weight
        entry = {
            "layer": i,
            "WQ": Wq.detach().cpu().clone(),
            "WK": Wk.detach().cpu().clone(),
            "WV": Wv.detach().cpu().clone(),
            "WO": attn.out_proj.weight.detach().cpu().clone(),
        }
        # MLP (feedforward) weights
        if hasattr(layer, 'linear1'):
            entry["W_up"]   = layer.linear1.weight.detach().cpu().clone()   # (d_ff, d_model)
            entry["W_down"] = layer.linear2.weight.detach().cpu().clone()   # (d_model, d_ff)
        logs.append(entry)
    return logs


def flatten_model_params(model):
    """Flatten all model parameters into a single 1-D tensor."""
    return torch.cat([p.detach().cpu().reshape(-1) for p in model.parameters()])


# ═══════════════════════════════════════════════════════════════════════════
# Single run
# ═══════════════════════════════════════════════════════════════════════════

def run_single(cfg: DyckSweepConfig):
    device = get_device()

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    # Build fixed dataset (same across all seeds via DATA_SEED)
    X_all, Y_all = build_depth_dataset(
        n_seqs=cfg.N_TOTAL, max_pairs=cfg.MAX_PAIRS,
        ctx_len=cfg.CTX_LEN, seed=cfg.DATA_SEED
    )
    frac = cfg.N_TRAIN / cfg.N_TOTAL
    train_x, train_y, test_x, test_y = split_dataset(
        X_all, Y_all, frac_train=frac, seed=cfg.DATA_SEED
    )
    train_toks = (train_y != -100).sum().item()
    test_toks = (test_y != -100).sum().item()
    print(f"    Dataset: {len(train_x)} train ({train_toks} toks), "
          f"{len(test_x)} test ({test_toks} toks)")

    model = DyckTransformerLM(
        vocab_size=VOCAB_SIZE,
        ctx_len=max(cfg.CTX_LEN, cfg.CTX_LEN_OOD),
        d_model=cfg.D_MODEL,
        n_layers=cfg.N_LAYERS,
        n_heads=cfg.N_HEADS,
        d_ff=cfg.D_FF,
        dropout=cfg.DROPOUT,
        n_classes=cfg.N_CLASSES,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Model: {n_params:,} params (overparameterized {n_params/train_toks:.0f}x), "
          f"device={device}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2),
    )

    attn_logs = [{"step": 0, "layers": extract_attn_matrices(model)}]
    metrics = []
    patience = 0
    t0 = time.time()
    grokked = False

    for step in range(1, cfg.STEPS + 1):
        model.train()
        # Sample batch from fixed training set
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

            # OOD eval
            ood_rng = random.Random(cfg.SEED + 777 + step)
            ood_loss, ood_acc = eval_loss_acc(
                model, cfg.EVAL_BATCHES_OOD, cfg.EVAL_BS_OOD,
                cfg.CTX_LEN_OOD, cfg.MAX_PAIRS_OOD, device, ood_rng)

            metrics.append({
                "step": step,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "ood_loss": ood_loss,
                "ood_acc": ood_acc,
            })

            if step % (cfg.EVAL_EVERY * 5) == 0 or step == 1:
                elapsed = (time.time() - t0) / 60
                wd_tag = f"wd={cfg.WEIGHT_DECAY}"
                print(f"  [s{cfg.SEED} {wd_tag}] "
                      f"step {step:6d} | "
                      f"train {train_loss:.4f}/{train_acc:.3f} | "
                      f"test {test_loss:.4f}/{test_acc:.3f} | "
                      f"ood {ood_loss:.4f}/{ood_acc:.3f} | "
                      f"{elapsed:.1f}m")

            if test_acc >= cfg.STOP_ACC:
                patience += 1
                if patience >= cfg.STOP_PATIENCE:
                    grokked = True
                    print(f"  >>> GROKKED at step {step} (test_acc={test_acc:.4f})")
                    break
            else:
                patience = 0

    result = {
        "attn_logs": attn_logs,
        "cfg": asdict(cfg),
        "metrics": metrics,
        "grokked": grokked,
        "final_step": step,
        "final_train_loss": metrics[-1]["train_loss"] if metrics else 0,
        "final_train_acc": metrics[-1]["train_acc"] if metrics else 0,
        "final_test_loss": metrics[-1]["test_loss"] if metrics else 0,
        "final_test_acc": metrics[-1]["test_acc"] if metrics else 0,
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Sweep
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)

    seeds = [42, 137, 2024]
    weight_decays = [1.0, 0.0]

    total = len(weight_decays) * len(seeds)
    print(f"Sweep: {len(weight_decays)} wd x {len(seeds)} seeds = {total} runs")
    print(f"Output: {OUT_DIR}/")
    print()

    summary = []
    run_idx = 0

    for wd in weight_decays:
        for seed in seeds:
            run_idx += 1
            tag = f"dyck_wd{wd}_s{seed}"
            out_path = OUT_DIR / f"{tag}.pt"

            if out_path.exists():
                print(f"[{run_idx}/{total}] {tag} -- already exists, skipping")
                data = torch.load(out_path, map_location="cpu", weights_only=False)
                summary.append({
                    "wd": wd, "seed": seed,
                    "grokked": data["grokked"],
                    "final_step": data["final_step"],
                    "final_test_loss": data.get("final_test_loss", 0),
                    "final_test_acc": data.get("final_test_acc", 0),
                    "n_snapshots": len(data["attn_logs"]),
                })
                continue

            print(f"\n[{run_idx}/{total}] {tag}")
            if wd == 0.0:
                cfg = DyckSweepConfig(WEIGHT_DECAY=wd, SEED=seed,
                                      STEPS=10_000, MODEL_LOG_EVERY=500)
            else:
                cfg = DyckSweepConfig(WEIGHT_DECAY=wd, SEED=seed)
            result = run_single(cfg)

            torch.save(result, out_path)
            print(f"  saved -> {out_path.name} "
                  f"({len(result['attn_logs'])} snaps, grokked={result['grokked']})")

            summary.append({
                "wd": wd, "seed": seed,
                "grokked": result["grokked"],
                "final_step": result["final_step"],
                "final_test_loss": result["final_test_loss"],
                "final_test_acc": result["final_test_acc"],
                "n_snapshots": len(result["attn_logs"]),
            })

    # save summary
    summary_path = OUT_DIR / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"\n{'wd':>4s}  {'seed':>5s}  {'grok':>5s}  {'step':>6s}  "
          f"{'test_loss':>9s}  {'test_acc':>8s}  {'snaps':>5s}")
    print(f"{'---':>4s}  {'---':>5s}  {'---':>5s}  {'---':>6s}  "
          f"{'---':>9s}  {'---':>8s}  {'---':>5s}")
    for s in summary:
        print(f"{s['wd']:4.1f}  {s['seed']:5d}  "
              f"{'YES' if s['grokked'] else 'no':>5s}  {s['final_step']:6d}  "
              f"{s['final_test_loss']:9.4f}  {s['final_test_acc']:8.3f}  "
              f"{s['n_snapshots']:5d}")

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
