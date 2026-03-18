#!/usr/bin/env python3
"""
SCAN seq2seq pilot with geometry probes.

Equivalent of dyck_lm_pilot.py adapted for the SCAN command-to-action task.

Architecture: 4-layer, 256-dim encoder-decoder Transformer (ScanTransformer)
Geometry probes: gradient cosine alignment, deltaW tracking (same as dyck_lm_pilot.py)

Logs to pilot_log.jsonl every EVAL_INTERVAL steps.
"""

import math
import json
import time
import random
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 1) SCAN data download and parsing (from scan_grok_sweep.py)
# ═══════════════════════════════════════════════════════════════════════════

SCAN_DATA_DIR = Path(__file__).parent / "../scan_data"
SCAN_URLS = {
    "train": "https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_train_simple.txt",
    "test": "https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_test_simple.txt",
}


def download_scan():
    """Download SCAN simple split from GitHub."""
    SCAN_DATA_DIR.mkdir(exist_ok=True)
    for split, url in SCAN_URLS.items():
        out_path = SCAN_DATA_DIR / f"tasks_{split}_simple.txt"
        if out_path.exists():
            continue
        print(f"  Downloading SCAN {split} split...")
        urllib.request.urlretrieve(url, out_path)
        print(f"    saved -> {out_path}")


def parse_scan_file(filepath):
    """Parse SCAN file. Each line: 'IN: <command> OUT: <actions>'"""
    commands, actions = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" OUT: ")
            cmd = parts[0].replace("IN: ", "")
            act = parts[1]
            commands.append(cmd.split())
            actions.append(act.split())
    return commands, actions


# ═══════════════════════════════════════════════════════════════════════════
# 2) Vocabulary and tokenization (from scan_grok_sweep.py)
# ═══════════════════════════════════════════════════════════════════════════

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"


class Vocab:
    """Simple vocabulary for SCAN commands/actions."""
    def __init__(self):
        self.token2idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}
        self.idx2token = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN}
        self.size = 3

    def add(self, token):
        if token not in self.token2idx:
            idx = self.size
            self.token2idx[token] = idx
            self.idx2token[idx] = token
            self.size += 1

    def encode(self, tokens, add_sos=False, add_eos=False):
        ids = []
        if add_sos:
            ids.append(self.token2idx[SOS_TOKEN])
        for t in tokens:
            ids.append(self.token2idx[t])
        if add_eos:
            ids.append(self.token2idx[EOS_TOKEN])
        return ids

    def decode(self, ids):
        return [self.idx2token[i] for i in ids
                if i not in (self.token2idx[PAD_TOKEN],
                             self.token2idx[SOS_TOKEN],
                             self.token2idx[EOS_TOKEN])]


def build_vocabs(train_cmds, train_acts, test_cmds=None, test_acts=None):
    """Build command and action vocabularies from training data."""
    cmd_vocab = Vocab()
    act_vocab = Vocab()
    for cmd in train_cmds:
        for t in cmd:
            cmd_vocab.add(t)
    for act in train_acts:
        for t in act:
            act_vocab.add(t)
    if test_cmds is not None:
        for cmd in test_cmds:
            for t in cmd:
                cmd_vocab.add(t)
    if test_acts is not None:
        for act in test_acts:
            for t in act:
                act_vocab.add(t)
    return cmd_vocab, act_vocab


def build_scan_dataset(cmd_vocab, act_vocab, commands, actions, max_cmd_len, max_act_len):
    """
    Convert commands/actions to padded tensors.
    Returns (cmd_ids, act_input_ids, act_target_ids).
    cmd_ids: [N, max_cmd_len]          -- encoder input
    act_input_ids: [N, max_act_len]    -- decoder input (SOS + actions)
    act_target_ids: [N, max_act_len]   -- decoder target (actions + EOS), -100 for padding
    """
    N = len(commands)
    cmd_ids = torch.full((N, max_cmd_len), cmd_vocab.token2idx[PAD_TOKEN], dtype=torch.long)
    act_in = torch.full((N, max_act_len), act_vocab.token2idx[PAD_TOKEN], dtype=torch.long)
    act_tgt = torch.full((N, max_act_len), -100, dtype=torch.long)

    for i in range(N):
        c_enc = cmd_vocab.encode(commands[i])
        clen = min(len(c_enc), max_cmd_len)
        cmd_ids[i, :clen] = torch.tensor(c_enc[:clen])

        a_in = act_vocab.encode(actions[i], add_sos=True)
        a_tgt = act_vocab.encode(actions[i], add_eos=True)

        alen = min(len(a_in), max_act_len)
        act_in[i, :alen] = torch.tensor(a_in[:alen])
        act_tgt[i, :alen] = torch.tensor(a_tgt[:alen])

    return cmd_ids, act_in, act_tgt


# ═══════════════════════════════════════════════════════════════════════════
# 3) Encoder-decoder Transformer (ScanTransformer from scan_grok_sweep.py)
# ═══════════════════════════════════════════════════════════════════════════

class ScanTransformer(nn.Module):
    """
    Encoder-decoder Transformer for SCAN command-to-action translation.
    Uses nn.Transformer with causal decoder mask.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, max_src_len, max_tgt_len,
                 d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Encoder embeddings
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.src_pos_emb = nn.Embedding(max_src_len, d_model)

        # Decoder embeddings
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.tgt_pos_emb = nn.Embedding(max_tgt_len, d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=n_heads,
            num_encoder_layers=n_layers, num_decoder_layers=n_layers,
            dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None):
        """
        src: [B, S] -- source token IDs
        tgt: [B, T] -- target token IDs (decoder input)
        """
        B, S = src.shape
        _, T = tgt.shape

        # Encoder
        src_pos = torch.arange(S, device=src.device)
        src_emb = self.src_tok_emb(src) + self.src_pos_emb(src_pos)[None, :, :]

        # Decoder
        tgt_pos = torch.arange(T, device=tgt.device)
        tgt_emb = self.tgt_tok_emb(tgt) + self.tgt_pos_emb(tgt_pos)[None, :, :]

        # Causal mask for decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=tgt.device)

        h = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_is_causal=True,
        )
        h = self.ln(h)
        return self.head(h)  # [B, T, tgt_vocab_size]


# ═══════════════════════════════════════════════════════════════════════════
# 4) Loss and accuracy helpers
# ═══════════════════════════════════════════════════════════════════════════

def masked_ce_loss(logits, targets):
    """Cross-entropy ignoring -100 targets (padding)."""
    B, T, V = logits.shape
    logits_flat = logits.view(B * T, V)
    targets_flat = targets.view(B * T)
    mask = targets_flat != -100
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.cross_entropy(logits_flat[mask], targets_flat[mask])


def masked_accuracy(logits, targets):
    """Per-token accuracy ignoring padding (-100)."""
    preds = logits.argmax(dim=-1)
    mask = targets != -100
    if mask.sum() == 0:
        return 0.0
    correct = ((preds == targets) & mask).sum().item()
    return correct / mask.sum().item()


def sequence_accuracy(logits, targets):
    """Full-sequence accuracy (all tokens must match)."""
    preds = logits.argmax(dim=-1)
    mask = targets != -100
    B = logits.shape[0]
    correct = 0
    for i in range(B):
        m = mask[i]
        if m.sum() == 0:
            continue
        if (preds[i][m] == targets[i][m]).all():
            correct += 1
    return correct / B


# ═══════════════════════════════════════════════════════════════════════════
# 5) Batch sampling helper
# ═══════════════════════════════════════════════════════════════════════════

def sample_batch(cmd_ids, act_in, act_tgt, batch_size, device):
    """Sample a random batch from a fixed dataset."""
    idx = torch.randint(0, len(cmd_ids), (batch_size,))
    return (cmd_ids[idx].to(device), act_in[idx].to(device),
            act_tgt[idx].to(device))


# ═══════════════════════════════════════════════════════════════════════════
# 6) Eval helper
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_on_dataset(model, cmd_ids, act_in, act_tgt, device, pad_id=0,
                    batch_size=256):
    """Evaluate loss, token accuracy, and sequence accuracy on a dataset."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_seq_correct = 0
    total_seqs = 0
    n = len(cmd_ids)

    for i in range(0, n, batch_size):
        src = cmd_ids[i:i+batch_size].to(device)
        tgt_in = act_in[i:i+batch_size].to(device)
        tgt_out = act_tgt[i:i+batch_size].to(device)

        src_pad_mask = (src == pad_id)
        tgt_pad_mask = (tgt_in == pad_id)

        logits = model(src, tgt_in, src_pad_mask=src_pad_mask,
                       tgt_pad_mask=tgt_pad_mask)
        loss = masked_ce_loss(logits, tgt_out)

        mask = tgt_out != -100
        preds = logits.argmax(dim=-1)
        total_correct += ((preds == tgt_out) & mask).sum().item()
        total_tokens += mask.sum().item()
        total_loss += loss.item() * mask.sum().item()

        B = src.shape[0]
        for j in range(B):
            m = mask[j]
            if m.sum() > 0 and (preds[j][m] == tgt_out[j][m]).all():
                total_seq_correct += 1
            total_seqs += 1

    avg_loss = total_loss / max(total_tokens, 1)
    tok_acc = total_correct / max(total_tokens, 1)
    seq_acc = total_seq_correct / max(total_seqs, 1)
    model.train()
    return avg_loss, tok_acc, seq_acc


# ═══════════════════════════════════════════════════════════════════════════
# 7) Geometry probes (adapted from dyck_lm_pilot.py for seq2seq)
# ═══════════════════════════════════════════════════════════════════════════

def get_param_by_name(model: nn.Module, name: str) -> torch.Tensor:
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise KeyError(name)


def cosine_abs(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.flatten()
    b = b.flatten()
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return float((a @ b).abs() / denom)


def random_baseline_abs_cos(dim: int, n_samples: int = 256, device="cpu", seed: int = 0) -> float:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    vals = []
    for _ in range(n_samples):
        a = torch.randn(dim, generator=g, device=device)
        b = torch.randn(dim, generator=g, device=device)
        vals.append(cosine_abs(a, b))
    return sum(vals) / len(vals)


def grad_on_probe_seq2seq(model, src, tgt_in, tgt_out, param_name, pad_id_src, pad_id_tgt):
    """
    Compute gradient of masked CE loss w.r.t. a named parameter on a seq2seq batch.
    """
    model.zero_grad(set_to_none=True)
    src_pad_mask = (src == pad_id_src)
    tgt_pad_mask = (tgt_in == pad_id_tgt)
    logits = model(src, tgt_in, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
    loss = masked_ce_loss(logits, tgt_out)
    loss.backward()
    p = get_param_by_name(model, param_name)
    assert p.grad is not None
    return p.grad.detach().clone()


# ═══════════════════════════════════════════════════════════════════════════
# 8) Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    seed: int = 42
    device: str = "mps"  # or "cuda" / "cpu"
    steps: int = 50_000
    batch_size: int = 256

    # Model (4-layer, 256-dim encoder-decoder)
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0

    # Optimizer
    lr: float = 1e-4
    wd: float = 1.0
    betas: Tuple[float, float] = (0.9, 0.98)
    grad_clip: float = 1.0

    # Data
    n_train: int = 2048
    n_test_eval: int = 500
    data_seed: int = 0

    # Logging
    log_path: str = "pilot_log.jsonl"
    eval_interval: int = 500
    eval_bs: int = 256

    # Geometry probe
    probe_param: str = "transformer.encoder.layers.0.self_attn.in_proj_weight"
    probe_batch: int = 256
    delta_steps: int = 500
    baseline_cos_seed: int = 123


# ═══════════════════════════════════════════════════════════════════════════
# 9) Device helper
# ═══════════════════════════════════════════════════════════════════════════

def get_device(requested: str) -> torch.device:
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested in ("mps", "cuda"):
        return torch.device("cpu")
    return torch.device(requested)


# ═══════════════════════════════════════════════════════════════════════════
# 10) Main training loop
# ═══════════════════════════════════════════════════════════════════════════

def main(cfg: Config):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = get_device(cfg.device)
    print(f"Device: {device}")

    # --- Download and load SCAN data ---
    download_scan()

    train_cmds, train_acts = parse_scan_file(
        SCAN_DATA_DIR / "tasks_train_simple.txt")
    test_cmds, test_acts = parse_scan_file(
        SCAN_DATA_DIR / "tasks_test_simple.txt")

    # Subsample training set
    rng = random.Random(cfg.data_seed)
    indices = list(range(len(train_cmds)))
    rng.shuffle(indices)
    n_train = min(cfg.n_train, len(train_cmds))
    train_indices = indices[:n_train]
    sub_train_cmds = [train_cmds[i] for i in train_indices]
    sub_train_acts = [train_acts[i] for i in train_indices]

    # Build vocabularies
    cmd_vocab, act_vocab = build_vocabs(train_cmds, train_acts,
                                        test_cmds, test_acts)
    print(f"Vocab: {cmd_vocab.size} commands, {act_vocab.size} actions")

    # Compute max lengths
    all_cmds = sub_train_cmds + test_cmds
    all_acts = sub_train_acts + test_acts
    max_cmd_len = max(len(c) for c in all_cmds) + 2
    max_act_len = max(len(a) for a in all_acts) + 2  # +2 for SOS/EOS

    # Build datasets
    train_src, train_tgt_in, train_tgt_out = build_scan_dataset(
        cmd_vocab, act_vocab, sub_train_cmds, sub_train_acts,
        max_cmd_len, max_act_len)

    test_src, test_tgt_in, test_tgt_out = build_scan_dataset(
        cmd_vocab, act_vocab, test_cmds, test_acts,
        max_cmd_len, max_act_len)

    # Subsample test set for fast evaluation
    n_test_eval = min(cfg.n_test_eval, len(test_src))
    test_perm = torch.randperm(len(test_src),
                               generator=torch.Generator().manual_seed(cfg.data_seed))
    eval_test_src = test_src[test_perm[:n_test_eval]]
    eval_test_tgt_in = test_tgt_in[test_perm[:n_test_eval]]
    eval_test_tgt_out = test_tgt_out[test_perm[:n_test_eval]]

    train_toks = (train_tgt_out != -100).sum().item()
    test_toks = (eval_test_tgt_out != -100).sum().item()
    print(f"Dataset: {len(train_src)} train ({train_toks} toks), "
          f"{n_test_eval} test-eval ({test_toks} toks) from {len(test_src)} total")
    print(f"Max lengths: cmd={max_cmd_len}, act={max_act_len}")

    # --- Build model ---
    model = ScanTransformer(
        src_vocab_size=cmd_vocab.size,
        tgt_vocab_size=act_vocab.size,
        max_src_len=max_cmd_len,
        max_tgt_len=max_act_len,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params (overparameterized {n_params/train_toks:.0f}x), "
          f"device={device}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.wd,
        betas=cfg.betas,
    )

    pad_id_src = cmd_vocab.token2idx[PAD_TOKEN]
    pad_id_tgt = act_vocab.token2idx[PAD_TOKEN]

    # --- Geometry probe setup ---
    p0 = get_param_by_name(model, cfg.probe_param)
    # Random baseline for absolute cosine similarity in this dimension
    baseline = random_baseline_abs_cos(p0.numel(), device=str(device),
                                       seed=cfg.baseline_cos_seed)
    print(f"Probe param: {cfg.probe_param} ({p0.numel():,} elements)")
    print(f"Random baseline |cos|: {baseline:.6f}")

    # For delta-W tracking
    last_probe_weight = p0.detach().clone()
    last_probe_step = 0

    def log_row(row: Dict):
        with open(cfg.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    # --- Training loop ---
    t0 = time.time()
    print(f"\nStarting training: {cfg.steps} steps, bs={cfg.batch_size}, "
          f"lr={cfg.lr}, wd={cfg.wd}")
    print(f"Logging to {cfg.log_path} every {cfg.eval_interval} steps\n")

    for step in range(1, cfg.steps + 1):
        model.train()
        src, tgt_in, tgt_out = sample_batch(
            train_src, train_tgt_in, train_tgt_out, cfg.batch_size, device)

        src_pad_mask = (src == pad_id_src)
        tgt_pad_mask = (tgt_in == pad_id_tgt)

        logits = model(src, tgt_in, src_pad_mask=src_pad_mask,
                       tgt_pad_mask=tgt_pad_mask)
        loss = masked_ce_loss(logits, tgt_out)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        # --- Eval + geometry probes ---
        if step % cfg.eval_interval == 0 or step == 1:
            # Evaluate on train and test sets
            train_loss, train_tok_acc, train_seq_acc = eval_on_dataset(
                model, train_src, train_tgt_in, train_tgt_out, device,
                pad_id=pad_id_src, batch_size=cfg.eval_bs)
            test_loss, test_tok_acc, test_seq_acc = eval_on_dataset(
                model, eval_test_src, eval_test_tgt_in, eval_test_tgt_out,
                device, pad_id=pad_id_src, batch_size=cfg.eval_bs)

            # --- Geometry probes on two independent mini-batches ---
            src_b1, tgt_in_b1, tgt_out_b1 = sample_batch(
                train_src, train_tgt_in, train_tgt_out, cfg.probe_batch, device)
            src_b2, tgt_in_b2, tgt_out_b2 = sample_batch(
                train_src, train_tgt_in, train_tgt_out, cfg.probe_batch, device)

            g1 = grad_on_probe_seq2seq(model, src_b1, tgt_in_b1, tgt_out_b1,
                                        cfg.probe_param, pad_id_src, pad_id_tgt)
            g2 = grad_on_probe_seq2seq(model, src_b2, tgt_in_b2, tgt_out_b2,
                                        cfg.probe_param, pad_id_src, pad_id_tgt)
            delta_grad = g1 - g2
            defect_proxy = float(delta_grad.norm().item())

            # delta-W over delta_steps
            p = get_param_by_name(model, cfg.probe_param)
            if step - last_probe_step >= cfg.delta_steps:
                dW = p.detach() - last_probe_weight
                align = cosine_abs(dW, delta_grad)
                align_ratio = float(align / baseline) if baseline > 0 else float("nan")

                last_probe_weight = p.detach().clone()
                last_probe_step = step
            else:
                align_ratio = None

            row = dict(
                step=step,
                lr=cfg.lr,
                wd=cfg.wd,
                seed=cfg.seed,
                train_loss=train_loss,
                train_tok_acc=train_tok_acc,
                train_seq_acc=train_seq_acc,
                test_loss=test_loss,
                test_tok_acc=test_tok_acc,
                test_seq_acc=test_seq_acc,
                defect_proxy=defect_proxy,
                align_ratio=align_ratio,
                baseline_abs_cos=baseline,
                seconds=time.time() - t0,
            )
            align_str = f"{align_ratio:.4f}" if align_ratio is not None else "---"
            print(f"step {step:6d} | "
                  f"train {train_loss:.4f}/{train_tok_acc:.3f}/{train_seq_acc:.3f} | "
                  f"test {test_loss:.4f}/{test_tok_acc:.3f}/{test_seq_acc:.3f} | "
                  f"defect {defect_proxy:.4f} | "
                  f"align {align_str}")
            log_row(row)

    elapsed = time.time() - t0
    print(f"\nDone. {cfg.steps} steps in {elapsed/60:.1f} min. Log: {cfg.log_path}")


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
