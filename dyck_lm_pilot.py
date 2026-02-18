import math
import json
import time
import random
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 1) Dyck-1 data generation
# -------------------------

TOK_OPEN = 0   # "("
TOK_CLOSE = 1  # ")"
TOK_PAD = 2    # optional pad
VOCAB_SIZE = 3

def gen_balanced_parentheses(max_pairs: int, rng: random.Random) -> List[int]:
    """
    Generate a balanced parentheses string with <= max_pairs pairs.
    Classic stack-walk: sample valid steps; end when pairs used and stack empty.
    """
    n_pairs = rng.randint(1, max_pairs)
    seq = []
    open_used = 0
    stack = 0
    # We generate exactly 2*n_pairs tokens
    for _ in range(2 * n_pairs):
        can_open = open_used < n_pairs
        can_close = stack > 0
        if can_open and can_close:
            # bias slightly toward opening early
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

def make_batch(
    batch_size: int,
    ctx_len: int,
    max_pairs: int,
    device: torch.device,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Next-token LM: given x[0..T-1], predict y[0..T-1] = x[1..T] (shifted).
    We pad sequences to ctx_len with PAD, and mask PAD loss.
    """
    x = torch.full((batch_size, ctx_len), TOK_PAD, dtype=torch.long)
    for i in range(batch_size):
        seq = gen_balanced_parentheses(max_pairs=max_pairs, rng=rng)
        seq = seq[:ctx_len]  # truncate if needed (rare)
        x[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    # y is next token; last target can be PAD
    y = x.clone()
    y[:, :-1] = x[:, 1:]
    y[:, -1] = TOK_PAD

    return x.to(device), y.to(device)

# -------------------------
# 2) Tiny Transformer LM
# -------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.Wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # B,h,T,hd
        k = self.Wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # B,h,T,T
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)

        out = att @ v  # B,h,T,hd
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.Wo(out)
        return out

class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, ctx_len: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ctx_len = ctx_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(ctx_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# -------------------------
# 3) Probes (cheap geometry)
# -------------------------

def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Ignore PAD targets
    B, T, V = logits.shape
    logits = logits.view(B*T, V)
    targets = targets.view(B*T)
    mask = targets != TOK_PAD
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.cross_entropy(logits[mask], targets[mask])

@torch.no_grad()
def eval_loss(model: nn.Module, steps: int, batch_size: int, ctx_len: int, max_pairs: int, device, rng: random.Random) -> float:
    model.eval()
    losses = []
    for _ in range(steps):
        x, y = make_batch(batch_size, ctx_len, max_pairs, device, rng)
        logits = model(x)
        loss = masked_ce_loss(logits, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

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

def grad_on_probe(model: nn.Module, x: torch.Tensor, y: torch.Tensor, param_name: str) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = masked_ce_loss(logits, y)
    loss.backward()
    p = get_param_by_name(model, param_name)
    assert p.grad is not None
    return p.grad.detach().clone()

# -------------------------
# 4) Training config + loop
# -------------------------

@dataclass
class Config:
    seed: int = 42
    device: str = "mps"  # or "cuda" / "cpu"
    steps: int = 50000
    batch_size: int = 64
    ctx_len_train: int = 64
    ctx_len_eval_short: int = 64
    ctx_len_eval_long: int = 256
    max_pairs_train: int = 32
    max_pairs_eval_short: int = 32
    max_pairs_eval_long: int = 128

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0

    lr: float = 3e-4
    wd: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)

    log_path: str = "pilot_log.jsonl"
    eval_interval: int = 500
    eval_batches: int = 20

    # geometry probe
    probe_param: str = "blocks.0.attn.Wq.weight"  # sentinel matrix
    probe_batch: int = 64
    probe_ctx: int = 64
    probe_pairs: int = 32
    delta_steps: int = 500  # Δt for ΔW
    baseline_cos_seed: int = 123

def main(cfg: Config):
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    device = torch.device(cfg.device if (cfg.device != "mps" or torch.backends.mps.is_available()) else "cpu")

    model = TinyLM(
        vocab_size=VOCAB_SIZE,
        ctx_len=max(cfg.ctx_len_eval_long, cfg.ctx_len_train, cfg.probe_ctx),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.wd)

    # Precompute random baseline for abs cosine for this parameter
    p0 = get_param_by_name(model, cfg.probe_param)
    baseline = random_baseline_abs_cos(p0.numel(), device=str(device), seed=cfg.baseline_cos_seed)

    # For ΔW
    last_probe_weight = p0.detach().clone()
    last_probe_step = 0

    def log_row(row: Dict):
        with open(cfg.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        x, y = make_batch(cfg.batch_size, cfg.ctx_len_train, cfg.max_pairs_train, device, rng)
        logits = model(x)
        loss = masked_ce_loss(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.eval_interval == 0 or step == 1:
            # losses
            eval_rng = random.Random(cfg.seed + 999 + step)
            train_loss = float(loss.item())
            val_short = eval_loss(model, cfg.eval_batches, cfg.batch_size, cfg.ctx_len_eval_short, cfg.max_pairs_eval_short, device, eval_rng)
            val_long = eval_loss(model, cfg.eval_batches, cfg.batch_size, cfg.ctx_len_eval_long, cfg.max_pairs_eval_long, device, eval_rng)

            # geometry probes on probe batches
            probe_rng = random.Random(cfg.seed + 12345 + step)
            xb1, yb1 = make_batch(cfg.probe_batch, cfg.probe_ctx, cfg.probe_pairs, device, probe_rng)
            xb2, yb2 = make_batch(cfg.probe_batch, cfg.probe_ctx, cfg.probe_pairs, device, probe_rng)

            g1 = grad_on_probe(model, xb1, yb1, cfg.probe_param)
            g2 = grad_on_probe(model, xb2, yb2, cfg.probe_param)
            delta_grad = (g1 - g2)
            defect_proxy = float(delta_grad.norm().item())

            # ΔW over delta_steps
            p = get_param_by_name(model, cfg.probe_param)
            if step - last_probe_step >= cfg.delta_steps:
                dW = (p.detach() - last_probe_weight)
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
                val_loss_short=val_short,
                val_loss_long=val_long,
                defect_proxy=defect_proxy,
                align_ratio=align_ratio,
                baseline_abs_cos=baseline,
                seconds=time.time() - t0,
            )
            print(row)
            log_row(row)

if __name__ == "__main__":
    cfg = Config()
    main(cfg)