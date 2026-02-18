#!/usr/bin/env python3
"""
SCAN grokking sweep for PCA eigenanalysis.

Replicates the experimental pipeline of the integrability-grokking paper
(Xu, 2026) on the SCAN (Simple Commands as Actions in Navigation) dataset.

Task:  Given a natural language command (e.g. "jump left twice"),
       predict the corresponding action sequence (e.g. "LTURN JUMP LTURN JUMP").
       This is a seq2seq deterministic map from command to actions.

"Grokking" setup:
  - Download SCAN simple split from GitHub.
  - Small training subset (~200 sequences) vs large test set.
  - The model first memorises the training set (100% train acc) while test
    accuracy stays lower; later, generalisation kicks in (grokking).
  - Weight decay is the critical control variable: WD>0 → grokking,
    WD=0 → memorisation only (test acc stalls or degrades).

Sweep dimensions:
  - Weight decay: {1.0, 0.0}
  - 3 random seeds

All results saved to scan_sweep_results/ as individual .pt files.
"""

import math, time, random, json, sys, os, urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# SCAN data download and parsing
# ═══════════════════════════════════════════════════════════════════════════

SCAN_DATA_DIR = Path(__file__).parent / "scan_data"
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
            # Format: "IN: walk OUT: WALK"
            parts = line.split(" OUT: ")
            cmd = parts[0].replace("IN: ", "")
            act = parts[1]
            commands.append(cmd.split())
            actions.append(act.split())
    return commands, actions


# ═══════════════════════════════════════════════════════════════════════════
# Vocabulary and tokenization
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
    # Also add test tokens to vocab (they share the same vocab in SCAN)
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
    cmd_ids: [N, max_cmd_len]          — encoder input
    act_input_ids: [N, max_act_len]    — decoder input (SOS + actions)
    act_target_ids: [N, max_act_len]   — decoder target (actions + EOS), -100 for padding
    """
    N = len(commands)
    cmd_ids = torch.full((N, max_cmd_len), cmd_vocab.token2idx[PAD_TOKEN], dtype=torch.long)
    act_in = torch.full((N, max_act_len), act_vocab.token2idx[PAD_TOKEN], dtype=torch.long)
    act_tgt = torch.full((N, max_act_len), -100, dtype=torch.long)

    for i in range(N):
        # Encode command
        c_enc = cmd_vocab.encode(commands[i])
        clen = min(len(c_enc), max_cmd_len)
        cmd_ids[i, :clen] = torch.tensor(c_enc[:clen])

        # Encode action: input = SOS + actions, target = actions + EOS
        a_in = act_vocab.encode(actions[i], add_sos=True)
        a_tgt = act_vocab.encode(actions[i], add_eos=True)

        alen = min(len(a_in), max_act_len)
        act_in[i, :alen] = torch.tensor(a_in[:alen])
        act_tgt[i, :alen] = torch.tensor(a_tgt[:alen])

    return cmd_ids, act_in, act_tgt


# ═══════════════════════════════════════════════════════════════════════════
# Transformer Seq2Seq model for SCAN
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
        src: [B, S] — source token IDs
        tgt: [B, T] — target token IDs (decoder input)
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
# Loss & accuracy helpers
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
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScanSweepConfig:
    # Model
    D_MODEL: int = 256
    N_LAYERS: int = 3
    N_HEADS: int = 4
    D_FF: int = 512
    DROPOUT: float = 0.0

    # Training
    LR: float = 1e-4
    BATCH_SIZE: int = 256
    STEPS: int = 30_000
    EVAL_EVERY: int = 500
    MODEL_LOG_EVERY: int = 500
    GRAD_CLIP: float = 1.0
    ADAM_BETA1: float = 0.9
    ADAM_BETA2: float = 0.98

    # Data — small subset for grokking
    N_TRAIN: int = 2048     # small enough to memorise, large enough to grok
    N_TEST_EVAL: int = 500  # subsample test set for fast eval
    DATA_SEED: int = 0      # fixed across all runs (same dataset subset)

    # Stopping
    STOP_ACC: float = 0.98  # stop if test_seq_acc >= this
    STOP_PATIENCE: int = 5

    # Sweep params (set per run)
    WEIGHT_DECAY: float = 1.0
    SEED: int = 42

    # Eval
    EVAL_BS: int = 256


OUT_DIR = Path(__file__).parent / "scan_sweep_results"


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

def sample_batch(cmd_ids, act_in, act_tgt, batch_size, device):
    """Sample a random batch from a fixed dataset."""
    idx = torch.randint(0, len(cmd_ids), (batch_size,))
    return (cmd_ids[idx].to(device), act_in[idx].to(device),
            act_tgt[idx].to(device))


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

        # Sequence accuracy
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


@torch.no_grad()
def extract_attn_matrices(model):
    """Extract attention + MLP weight matrices from all encoder + decoder layers."""
    logs = []

    # Encoder layers
    for i, layer in enumerate(model.transformer.encoder.layers):
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
            "type": "encoder",
            "WQ": Wq.detach().cpu().clone(),
            "WK": Wk.detach().cpu().clone(),
            "WV": Wv.detach().cpu().clone(),
            "WO": attn.out_proj.weight.detach().cpu().clone(),
        }
        if hasattr(layer, 'linear1'):
            entry["W_up"]   = layer.linear1.weight.detach().cpu().clone()
            entry["W_down"] = layer.linear2.weight.detach().cpu().clone()
        logs.append(entry)

    # Decoder layers
    for i, layer in enumerate(model.transformer.decoder.layers):
        # Self-attention
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
            "type": "decoder_self",
            "WQ": Wq.detach().cpu().clone(),
            "WK": Wk.detach().cpu().clone(),
            "WV": Wv.detach().cpu().clone(),
            "WO": attn.out_proj.weight.detach().cpu().clone(),
        }

        # Cross-attention
        xattn = layer.multihead_attn
        d2 = xattn.embed_dim
        if xattn._qkv_same_embed_dim:
            entry["XWQ"] = xattn.in_proj_weight[:d2].detach().cpu().clone()
            entry["XWK"] = xattn.in_proj_weight[d2:2*d2].detach().cpu().clone()
            entry["XWV"] = xattn.in_proj_weight[2*d2:].detach().cpu().clone()
        else:
            entry["XWQ"] = xattn.q_proj_weight.detach().cpu().clone()
            entry["XWK"] = xattn.k_proj_weight.detach().cpu().clone()
            entry["XWV"] = xattn.v_proj_weight.detach().cpu().clone()
        entry["XWO"] = xattn.out_proj.weight.detach().cpu().clone()

        # MLP
        if hasattr(layer, 'linear1'):
            entry["W_up"]   = layer.linear1.weight.detach().cpu().clone()
            entry["W_down"] = layer.linear2.weight.detach().cpu().clone()

        logs.append(entry)

    return logs


def flatten_model_params(model):
    """Flatten all model parameters into a single 1-D tensor."""
    return torch.cat([p.detach().cpu().reshape(-1) for p in model.parameters()])


# ═══════════════════════════════════════════════════════════════════════════
# Single run
# ═══════════════════════════════════════════════════════════════════════════

def run_single(cfg: ScanSweepConfig):
    device = get_device()

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    # Download SCAN
    download_scan()

    # Load full datasets
    train_cmds, train_acts = parse_scan_file(
        SCAN_DATA_DIR / "tasks_train_simple.txt")
    test_cmds, test_acts = parse_scan_file(
        SCAN_DATA_DIR / "tasks_test_simple.txt")

    # Subsample training set for grokking
    rng = random.Random(cfg.DATA_SEED)
    indices = list(range(len(train_cmds)))
    rng.shuffle(indices)
    n_train = min(cfg.N_TRAIN, len(train_cmds))
    train_indices = indices[:n_train]
    sub_train_cmds = [train_cmds[i] for i in train_indices]
    sub_train_acts = [train_acts[i] for i in train_indices]

    # Build vocabularies
    cmd_vocab, act_vocab = build_vocabs(train_cmds, train_acts,
                                         test_cmds, test_acts)
    print(f"    Vocab: {cmd_vocab.size} commands, {act_vocab.size} actions")

    # Compute max lengths
    all_cmds = sub_train_cmds + test_cmds
    all_acts = sub_train_acts + test_acts
    max_cmd_len = max(len(c) for c in all_cmds) + 2  # margin
    max_act_len = max(len(a) for a in all_acts) + 2  # +2 for SOS/EOS

    # Build datasets
    train_src, train_tgt_in, train_tgt_out = build_scan_dataset(
        cmd_vocab, act_vocab, sub_train_cmds, sub_train_acts,
        max_cmd_len, max_act_len)

    test_src, test_tgt_in, test_tgt_out = build_scan_dataset(
        cmd_vocab, act_vocab, test_cmds, test_acts,
        max_cmd_len, max_act_len)

    # Subsample test set for fast evaluation
    n_test_eval = min(cfg.N_TEST_EVAL, len(test_src))
    test_perm = torch.randperm(len(test_src),
                               generator=torch.Generator().manual_seed(cfg.DATA_SEED))
    eval_test_src = test_src[test_perm[:n_test_eval]]
    eval_test_tgt_in = test_tgt_in[test_perm[:n_test_eval]]
    eval_test_tgt_out = test_tgt_out[test_perm[:n_test_eval]]

    train_toks = (train_tgt_out != -100).sum().item()
    test_toks = (eval_test_tgt_out != -100).sum().item()
    print(f"    Dataset: {len(train_src)} train ({train_toks} toks), "
          f"{n_test_eval} test-eval ({test_toks} toks) from {len(test_src)} total")
    print(f"    Max lengths: cmd={max_cmd_len}, act={max_act_len}")

    model = ScanTransformer(
        src_vocab_size=cmd_vocab.size,
        tgt_vocab_size=act_vocab.size,
        max_src_len=max_cmd_len,
        max_tgt_len=max_act_len,
        d_model=cfg.D_MODEL,
        n_layers=cfg.N_LAYERS,
        n_heads=cfg.N_HEADS,
        d_ff=cfg.D_FF,
        dropout=cfg.DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Model: {n_params:,} params (overparameterized {n_params/train_toks:.0f}x), "
          f"device={device}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2),
    )

    pad_id = cmd_vocab.token2idx[PAD_TOKEN]
    attn_logs = [{"step": 0, "layers": extract_attn_matrices(model)}]
    metrics = []
    patience = 0
    t0 = time.time()
    grokked = False

    for step in range(1, cfg.STEPS + 1):
        model.train()
        src, tgt_in, tgt_out = sample_batch(
            train_src, train_tgt_in, train_tgt_out, cfg.BATCH_SIZE, device)

        src_pad_mask = (src == pad_id)
        tgt_pad_mask = (tgt_in == act_vocab.token2idx[PAD_TOKEN])

        logits = model(src, tgt_in, src_pad_mask=src_pad_mask,
                       tgt_pad_mask=tgt_pad_mask)
        loss = masked_ce_loss(logits, tgt_out)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        if step % cfg.MODEL_LOG_EVERY == 0:
            attn_logs.append({"step": step, "layers": extract_attn_matrices(model)})

        if step % cfg.EVAL_EVERY == 0 or step == 1:
            train_loss, train_tok_acc, train_seq_acc = eval_on_dataset(
                model, train_src, train_tgt_in, train_tgt_out, device,
                pad_id=pad_id, batch_size=cfg.EVAL_BS)
            test_loss, test_tok_acc, test_seq_acc = eval_on_dataset(
                model, eval_test_src, eval_test_tgt_in, eval_test_tgt_out,
                device, pad_id=pad_id, batch_size=cfg.EVAL_BS)

            metrics.append({
                "step": step,
                "train_loss": train_loss,
                "train_acc": train_tok_acc,
                "train_seq_acc": train_seq_acc,
                "test_loss": test_loss,
                "test_acc": test_tok_acc,
                "test_seq_acc": test_seq_acc,
            })

            if step % (cfg.EVAL_EVERY * 5) == 0 or step == 1:
                elapsed = (time.time() - t0) / 60
                wd_tag = f"wd={cfg.WEIGHT_DECAY}"
                print(f"  [s{cfg.SEED} {wd_tag}] "
                      f"step {step:6d} | "
                      f"train {train_loss:.4f}/{train_tok_acc:.3f}/{train_seq_acc:.3f} | "
                      f"test {test_loss:.4f}/{test_tok_acc:.3f}/{test_seq_acc:.3f} | "
                      f"{elapsed:.1f}m")

            if test_seq_acc >= cfg.STOP_ACC:
                patience += 1
                if patience >= cfg.STOP_PATIENCE:
                    grokked = True
                    print(f"  >>> GROKKED at step {step} "
                          f"(test_seq_acc={test_seq_acc:.4f})")
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
        "final_test_seq_acc": metrics[-1]["test_seq_acc"] if metrics else 0,
        "cmd_vocab_size": cmd_vocab.size,
        "act_vocab_size": act_vocab.size,
        "max_cmd_len": max_cmd_len,
        "max_act_len": max_act_len,
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
            tag = f"scan_wd{wd}_s{seed}"
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
                    "final_test_seq_acc": data.get("final_test_seq_acc", 0),
                    "n_snapshots": len(data["attn_logs"]),
                })
                continue

            print(f"\n[{run_idx}/{total}] {tag}")
            if wd == 0.0:
                cfg = ScanSweepConfig(WEIGHT_DECAY=wd, SEED=seed,
                                      STEPS=20_000, MODEL_LOG_EVERY=500)
            else:
                cfg = ScanSweepConfig(WEIGHT_DECAY=wd, SEED=seed)
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
                "final_test_seq_acc": result.get("final_test_seq_acc", 0),
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
          f"{'test_loss':>9s}  {'tok_acc':>8s}  {'seq_acc':>8s}  {'snaps':>5s}")
    print(f"{'---':>4s}  {'---':>5s}  {'---':>5s}  {'---':>6s}  "
          f"{'---':>9s}  {'---':>8s}  {'---':>8s}  {'---':>5s}")
    for s in summary:
        print(f"{s['wd']:4.1f}  {s['seed']:5d}  "
              f"{'YES' if s['grokked'] else 'no':>5s}  {s['final_step']:6d}  "
              f"{s['final_test_loss']:9.4f}  {s['final_test_acc']:8.3f}  "
              f"{s.get('final_test_seq_acc', 0):8.3f}  "
              f"{s['n_snapshots']:5d}")

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
