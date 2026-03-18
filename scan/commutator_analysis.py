#!/usr/bin/env python3
"""
Commutator defect analysis on the SCAN command-to-action manifold.

Adapts the commutator framework from the grokking codebase to the
SCAN seq2seq task.

Three-basis exec/random ratio comparison:
  1. Weight SVD    — top-k SVD of current weight W
  2. DeltaW SVD    — top-k SVD of (W_current - W_init)
  3. Gradient SVD  — top-k SVD of accumulated recent gradients

For each (wd, seed) condition:
  1. Train model, accumulating gradients and saving checkpoints
  2. At each checkpoint, compute commutator defect (K=COMM_K median)
  3. For each of 3 basis types, build joint basis and project commutator
  4. Compare exec projection vs random subspace projection -> ratio

Produces:
  figJ  -- Commutator defect over training (wd=1.0 vs wd=0.0)
  figK  -- Projected vs residual commutator (integrability, PCA traj basis)
  figM  -- Defect x integrability combined
  figN  -- Attention weight fraction of commutator
  figO  -- Three-basis exec/random ratio over training (hero figure)
  figO2 -- Bar chart: wd=1.0 vs wd=0.0 for all 3 bases
  figO3 -- Exec vs random breakdown
"""

import math, time, random, sys, copy
from dataclasses import asdict
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from scan_grok_sweep import (
    ScanSweepConfig, ScanTransformer, masked_ce_loss,
    get_device, extract_attn_matrices, eval_on_dataset,
    download_scan, parse_scan_file, build_vocabs, build_scan_dataset,
    sample_batch, flatten_model_params, PAD_TOKEN, SCAN_DATA_DIR,
)
from scan_pca_analysis import pca_on_trajectory, collect_trajectory

# -- config -------------------------------------------------------------------
OUT_DIR = Path(__file__).parent / "scan_pca_plots"
SEEDS = [42, 137, 2024]
CHECKPOINT_EVERY = 200
COMM_K = 9
COMM_ETA = 1e-3
N_PCA_COMPONENTS = 2
SVD_TOPK = 3
N_RANDOM_TRIALS = 5
GRAD_WINDOW = 50

BASIS_TYPES = ["weight_svd", "delta_w_svd", "grad_svd"]
BASIS_COLORS = {
    "weight_svd": "#2ecc71",
    "delta_w_svd": "#3498db",
    "grad_svd": "#9b59b6",
}
BASIS_LABELS = {
    "weight_svd": "Weight SVD",
    "delta_w_svd": "DeltaW SVD",
    "grad_svd": "Grad SVD",
}


# =============================================================================
# Commutator functions
# =============================================================================

def _param_offsets(model):
    offsets = {}
    cursor = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        offsets[id(p)] = cursor
        cursor += p.numel()
    return offsets, cursor


def _write_params(model, theta):
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            p.copy_(theta[offset:offset+n].view_as(p))
            offset += n


def _flatten_grad(model):
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in model.parameters() if p.requires_grad
    ])


def _flatten_params(model):
    return torch.cat([p.detach().flatten()
                      for p in model.parameters() if p.requires_grad])


def commutator_defect(model, batch_fn, device, eta=1e-3, eps=1e-12,
                      adaptive=True, min_perturb_norm=1e-6,
                      pad_id_src=0, pad_id_tgt=0):
    """Scale-normalized commutator defect for SCAN seq2seq.

    batch_fn returns (src, tgt_in, tgt_out) tuples.
    """
    was_training = model.training
    model.train()

    def flat_grad(src, tgt_in, tgt_out):
        model.zero_grad(set_to_none=True)
        src_pad_mask = (src == pad_id_src)
        tgt_pad_mask = (tgt_in == pad_id_tgt)
        logits = model(src, tgt_in,
                       src_pad_mask=src_pad_mask,
                       tgt_pad_mask=tgt_pad_mask)
        loss = masked_ce_loss(logits, tgt_out)
        loss.backward()
        return _flatten_grad(model)

    srcA, tgtA_in, tgtA_out = batch_fn()
    srcB, tgtB_in, tgtB_out = batch_fn()

    theta0 = _flatten_params(model)
    gA = flat_grad(srcA, tgtA_in, tgtA_out)
    gB = flat_grad(srcB, tgtB_in, tgtB_out)

    gA_norm = gA.norm()
    gB_norm = gB.norm()
    grad_cos = (gA @ gB) / (gA_norm * gB_norm + eps)

    effective_eta = eta
    if adaptive and gA_norm > 0 and gB_norm > 0:
        perturb_norm = min(eta * gA_norm.item(), eta * gB_norm.item())
        if perturb_norm < min_perturb_norm:
            effective_eta = min_perturb_norm / min(gA_norm.item(), gB_norm.item())
            effective_eta = min(effective_eta, 1.0)

    # Path AB
    _write_params(model, theta0 - effective_eta * gA)
    gB1 = flat_grad(srcB, tgtB_in, tgtB_out)
    thetaAB = theta0 - effective_eta * gA - effective_eta * gB1

    # Path BA
    _write_params(model, theta0 - effective_eta * gB)
    gA1 = flat_grad(srcA, tgtA_in, tgtA_out)
    thetaBA = theta0 - effective_eta * gB - effective_eta * gA1

    # Restore
    _write_params(model, theta0)
    if not was_training:
        model.eval()

    normA = (effective_eta * gA).norm()
    normB = (effective_eta * gB).norm()
    delta = thetaAB - thetaBA
    defect = (delta.norm() / (normA * normB + eps)).item()

    return defect, delta.detach(), grad_cos.item(), normA.detach(), normB.detach()


def projected_commutator(delta, B, normA, normB, eps=1e-12):
    delta = delta.reshape(-1)
    if B is None or delta.numel() != B.shape[0]:
        full_val = (delta.norm() / (normA * normB + eps)).item()
        return {"proj": float("nan"), "resid": float("nan"), "full": full_val}
    coeffs = B.T @ delta
    proj = B @ coeffs
    resid = delta - proj
    scale = normA * normB + eps
    return {
        "proj": (proj.norm() / scale).item(),
        "resid": (resid.norm() / scale).item(),
        "full": (delta.norm() / scale).item(),
    }


def random_projection_norm(delta, K, n_trials=5):
    P = delta.numel()
    delta_flat = delta.reshape(-1).cpu().float()
    results = []
    for _ in range(n_trials):
        G = torch.randn(P, K)
        Q, _ = torch.linalg.qr(G, mode="reduced")
        proj = Q @ (Q.T @ delta_flat)
        results.append(proj.norm().item())
    return results


# =============================================================================
# Block registry for encoder-decoder transformer
# =============================================================================

def get_block_registry(model):
    """Returns list of dicts describing each weight block."""
    offsets, total_params = _param_offsets(model)
    blocks = []

    # Encoder layers
    for li, layer in enumerate(model.transformer.encoder.layers):
        attn = layer.self_attn
        d = attn.embed_dim

        ip_id = id(attn.in_proj_weight)
        ip_off = offsets.get(ip_id, None)
        if ip_off is not None:
            for name, row_start in [("WQ", 0), ("WK", d), ("WV", 2*d)]:
                blocks.append({
                    "name": f"enc_L{li}_{name}",
                    "layer_type": "encoder",
                    "layer": li,
                    "offset": ip_off + row_start * d,
                    "shape": (d, d),
                    "numel": d * d,
                    "row_start": row_start,
                    "param_type": "in_proj",
                })

        out_id = id(attn.out_proj.weight)
        out_off = offsets.get(out_id, None)
        if out_off is not None:
            blocks.append({
                "name": f"enc_L{li}_WO",
                "layer_type": "encoder",
                "layer": li,
                "offset": out_off,
                "shape": attn.out_proj.weight.shape,
                "numel": attn.out_proj.weight.numel(),
                "row_start": None,
                "param_type": "out_proj",
            })

        if hasattr(layer, 'linear1'):
            l1_id = id(layer.linear1.weight)
            l1_off = offsets.get(l1_id, None)
            if l1_off is not None:
                blocks.append({
                    "name": f"enc_L{li}_MLP1",
                    "layer_type": "encoder",
                    "layer": li,
                    "offset": l1_off,
                    "shape": layer.linear1.weight.shape,
                    "numel": layer.linear1.weight.numel(),
                    "row_start": None,
                    "param_type": "mlp",
                })
            l2_id = id(layer.linear2.weight)
            l2_off = offsets.get(l2_id, None)
            if l2_off is not None:
                blocks.append({
                    "name": f"enc_L{li}_MLP2",
                    "layer_type": "encoder",
                    "layer": li,
                    "offset": l2_off,
                    "shape": layer.linear2.weight.shape,
                    "numel": layer.linear2.weight.numel(),
                    "row_start": None,
                    "param_type": "mlp",
                })

    # Decoder layers
    for li, layer in enumerate(model.transformer.decoder.layers):
        # Self attention
        attn = layer.self_attn
        d = attn.embed_dim

        ip_id = id(attn.in_proj_weight)
        ip_off = offsets.get(ip_id, None)
        if ip_off is not None:
            for name, row_start in [("WQ", 0), ("WK", d), ("WV", 2*d)]:
                blocks.append({
                    "name": f"dec_L{li}_{name}",
                    "layer_type": "decoder_self",
                    "layer": li,
                    "offset": ip_off + row_start * d,
                    "shape": (d, d),
                    "numel": d * d,
                    "row_start": row_start,
                    "param_type": "in_proj",
                })

        out_id = id(attn.out_proj.weight)
        out_off = offsets.get(out_id, None)
        if out_off is not None:
            blocks.append({
                "name": f"dec_L{li}_WO",
                "layer_type": "decoder_self",
                "layer": li,
                "offset": out_off,
                "shape": attn.out_proj.weight.shape,
                "numel": attn.out_proj.weight.numel(),
                "row_start": None,
                "param_type": "out_proj",
            })

        # Cross attention
        xattn = layer.multihead_attn
        d2 = xattn.embed_dim
        xip_id = id(xattn.in_proj_weight)
        xip_off = offsets.get(xip_id, None)
        if xip_off is not None:
            for name, row_start in [("XWQ", 0), ("XWK", d2), ("XWV", 2*d2)]:
                blocks.append({
                    "name": f"dec_L{li}_{name}",
                    "layer_type": "decoder_cross",
                    "layer": li,
                    "offset": xip_off + row_start * d2,
                    "shape": (d2, d2),
                    "numel": d2 * d2,
                    "row_start": row_start,
                    "param_type": "in_proj",
                })

        xout_id = id(xattn.out_proj.weight)
        xout_off = offsets.get(xout_id, None)
        if xout_off is not None:
            blocks.append({
                "name": f"dec_L{li}_XWO",
                "layer_type": "decoder_cross",
                "layer": li,
                "offset": xout_off,
                "shape": xattn.out_proj.weight.shape,
                "numel": xattn.out_proj.weight.numel(),
                "row_start": None,
                "param_type": "out_proj",
            })

        if hasattr(layer, 'linear1'):
            l1_id = id(layer.linear1.weight)
            l1_off = offsets.get(l1_id, None)
            if l1_off is not None:
                blocks.append({
                    "name": f"dec_L{li}_MLP1",
                    "layer_type": "decoder",
                    "layer": li,
                    "offset": l1_off,
                    "shape": layer.linear1.weight.shape,
                    "numel": layer.linear1.weight.numel(),
                    "row_start": None,
                    "param_type": "mlp",
                })
            l2_id = id(layer.linear2.weight)
            l2_off = offsets.get(l2_id, None)
            if l2_off is not None:
                blocks.append({
                    "name": f"dec_L{li}_MLP2",
                    "layer_type": "decoder",
                    "layer": li,
                    "offset": l2_off,
                    "shape": layer.linear2.weight.shape,
                    "numel": layer.linear2.weight.numel(),
                    "row_start": None,
                    "param_type": "mlp",
                })

    return blocks, total_params


def get_block_weight(model, block_info):
    """Extract current weight matrix for a block."""
    name = block_info["name"]
    d = model.d_model

    if name.startswith("enc_"):
        li = block_info["layer"]
        layer = model.transformer.encoder.layers[li]
        if "MLP1" in name:
            return layer.linear1.weight.detach()
        elif "MLP2" in name:
            return layer.linear2.weight.detach()
        elif "WO" in name:
            return layer.self_attn.out_proj.weight.detach()
        else:  # WQ, WK, WV
            ip_w = layer.self_attn.in_proj_weight.detach()
            row_start = block_info["row_start"]
            return ip_w[row_start:row_start+d, :]

    elif name.startswith("dec_"):
        li = block_info["layer"]
        layer = model.transformer.decoder.layers[li]
        if "MLP1" in name:
            return layer.linear1.weight.detach()
        elif "MLP2" in name:
            return layer.linear2.weight.detach()
        elif "XWO" in name:
            return layer.multihead_attn.out_proj.weight.detach()
        elif "XWQ" in name or "XWK" in name or "XWV" in name:
            ip_w = layer.multihead_attn.in_proj_weight.detach()
            row_start = block_info["row_start"]
            return ip_w[row_start:row_start+d, :]
        elif "WO" in name:
            return layer.self_attn.out_proj.weight.detach()
        else:  # WQ, WK, WV (self-attn)
            ip_w = layer.self_attn.in_proj_weight.detach()
            row_start = block_info["row_start"]
            return ip_w[row_start:row_start+d, :]

    return None


# =============================================================================
# Three basis constructors
# =============================================================================

def _block_basis(block, topk):
    U, S, Vh = torch.linalg.svd(block.float(), full_matrices=False)
    vecs = []
    r = min(topk, S.numel())
    for i in range(r):
        comp = S[i] * (U[:, i].unsqueeze(1) @ Vh[i].unsqueeze(0))
        flat = comp.reshape(-1)
        norm = flat.norm()
        if norm > 0:
            vecs.append(flat / norm)
    return vecs


def basis_weight_svd(model, block_info, k=3):
    W = get_block_weight(model, block_info)
    if W is None:
        return []
    return _block_basis(W, k)


def basis_delta_w_svd(model, block_info, init_weights, k=3):
    W = get_block_weight(model, block_info)
    if W is None:
        return []
    W0 = init_weights[block_info["name"]]
    delta_W = W.cpu() - W0.cpu()
    if delta_W.norm() < 1e-10:
        return _block_basis(W, k)
    return _block_basis(delta_W, k)


def basis_grad_svd(model, block_info, grad_accum, k=3):
    bname = block_info["name"]
    if bname not in grad_accum or grad_accum[bname] is None:
        W = get_block_weight(model, block_info)
        return _block_basis(W, k) if W is not None else []
    G = grad_accum[bname]
    if G.norm() < 1e-10:
        W = get_block_weight(model, block_info)
        return _block_basis(W, k) if W is not None else []
    return _block_basis(G, k)


# =============================================================================
# Joint basis + gradient accumulator
# =============================================================================

def build_joint_basis(model, blocks, total_params, basis_fn, k=3):
    basis_vecs = []
    for b in blocks:
        local_vecs = basis_fn(model, b, k=k)
        for vec in local_vecs:
            gv = torch.zeros(total_params)
            gv[b["offset"]:b["offset"] + b["numel"]] = vec.cpu()
            basis_vecs.append(gv)
    if not basis_vecs:
        return None
    B = torch.stack(basis_vecs, dim=1)
    B_ortho, _ = torch.linalg.qr(B, mode="reduced")
    return B_ortho


class GradAccumulator:
    """Maintains a running sum of recent gradients reshaped per block."""
    def __init__(self, blocks, model, window=GRAD_WINDOW):
        self.blocks = blocks
        self.model = model
        self.window = window
        self.buffer = []

    def push(self):
        grads = {}
        for b in self.blocks:
            g = self._get_block_grad(b)
            grads[b["name"]] = g.detach().cpu().clone() if g is not None else None
        self.buffer.append(grads)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)

    def _get_block_grad(self, block_info):
        name = block_info["name"]
        d = self.model.d_model

        if name.startswith("enc_"):
            li = block_info["layer"]
            layer = self.model.transformer.encoder.layers[li]
        elif name.startswith("dec_"):
            li = block_info["layer"]
            layer = self.model.transformer.decoder.layers[li]
        else:
            return None

        if "MLP1" in name:
            p = layer.linear1.weight
            return p.grad if p.grad is not None else None
        elif "MLP2" in name:
            p = layer.linear2.weight
            return p.grad if p.grad is not None else None
        elif "XWO" in name:
            p = layer.multihead_attn.out_proj.weight
            return p.grad if p.grad is not None else None
        elif "XWQ" in name or "XWK" in name or "XWV" in name:
            p = layer.multihead_attn.in_proj_weight
            if p.grad is not None:
                row_start = block_info["row_start"]
                return p.grad[row_start:row_start+d, :]
            return None
        elif "WO" in name:
            if name.startswith("enc_"):
                p = layer.self_attn.out_proj.weight
            else:
                p = layer.self_attn.out_proj.weight
            return p.grad if p.grad is not None else None
        else:  # WQ, WK, WV
            p = layer.self_attn.in_proj_weight
            if p.grad is not None:
                row_start = block_info["row_start"]
                return p.grad[row_start:row_start+d, :]
            return None

    def get_accum(self):
        result = {}
        for b in self.blocks:
            bname = b["name"]
            accum = None
            for grads in self.buffer:
                g = grads.get(bname, None)
                if g is not None:
                    if accum is None:
                        accum = g.clone()
                    else:
                        accum += g
            result[bname] = accum
        return result


# =============================================================================
# Masks for attn/mlp fraction
# =============================================================================

def attn_weight_mask(model):
    offsets, total_params = _param_offsets(model)
    mask = torch.zeros(total_params, dtype=torch.bool)
    # Encoder
    for layer in model.transformer.encoder.layers:
        attn_mod = layer.self_attn
        for p in [attn_mod.in_proj_weight, attn_mod.out_proj.weight]:
            if p.requires_grad and id(p) in offsets:
                start = offsets[id(p)]
                mask[start:start + p.numel()] = True
        for p in [attn_mod.in_proj_bias, attn_mod.out_proj.bias]:
            if p is not None and p.requires_grad and id(p) in offsets:
                start = offsets[id(p)]
                mask[start:start + p.numel()] = True
    # Decoder (self + cross attn)
    for layer in model.transformer.decoder.layers:
        for attn_mod in [layer.self_attn, layer.multihead_attn]:
            for p in [attn_mod.in_proj_weight, attn_mod.out_proj.weight]:
                if p.requires_grad and id(p) in offsets:
                    start = offsets[id(p)]
                    mask[start:start + p.numel()] = True
            for p in [attn_mod.in_proj_bias, attn_mod.out_proj.bias]:
                if p is not None and p.requires_grad and id(p) in offsets:
                    start = offsets[id(p)]
                    mask[start:start + p.numel()] = True
    return mask


def mlp_weight_mask(model):
    offsets, total_params = _param_offsets(model)
    mask = torch.zeros(total_params, dtype=torch.bool)
    for layers in [model.transformer.encoder.layers,
                   model.transformer.decoder.layers]:
        for layer in layers:
            for attr in ['linear1', 'linear2']:
                if not hasattr(layer, attr):
                    continue
                mod = getattr(layer, attr)
                for p in mod.parameters():
                    if p.requires_grad and id(p) in offsets:
                        start = offsets[id(p)]
                        mask[start:start + p.numel()] = True
    return mask


# =============================================================================
# Phase-based training
# =============================================================================

PHASE_NAMES = ["early", "memorization", "pre-grok", "post-grok"]


def _load_scan_data(cfg):
    """Load SCAN data and build vocabs/tensors."""
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


def train_and_measure_phases(cfg, checkpoint_every=200):
    """Train model, identify 4 phase points, do three-basis measurement."""
    device = get_device()

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

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

    model = ScanTransformer(
        src_vocab_size=cmd_vocab.size,
        tgt_vocab_size=act_vocab.size,
        max_src_len=data["max_cmd_len"],
        max_tgt_len=data["max_act_len"],
        d_model=cfg.D_MODEL, n_layers=cfg.N_LAYERS,
        n_heads=cfg.N_HEADS, d_ff=cfg.D_FF, dropout=cfg.DROPOUT,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    )

    blocks, total_params = get_block_registry(model)
    block_names = [b["name"] for b in blocks]
    print(f"    P={total_params}, {len(blocks)} blocks")

    init_weights = {}
    for b in blocks:
        init_weights[b["name"]] = get_block_weight(model, b).cpu().clone()

    grad_accum = GradAccumulator(blocks, model, window=GRAD_WINDOW)
    amask = attn_weight_mask(model)
    mmask = mlp_weight_mask(model)

    def batch_fn():
        return sample_batch(train_src, train_tgt_in, train_tgt_out,
                           cfg.BATCH_SIZE, device)

    attn_logs = [{"step": 0, "layers": extract_attn_matrices(model)}]
    checkpoints = [(0, {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    {k: (v.cpu().clone() if v is not None else None)
                     for k, v in grad_accum.get_accum().items()})]
    metrics = []
    defect_trace = []
    patience = 0
    grokked = False
    grok_step = None
    t0 = time.time()

    # Quick defect at step 0
    D0, delta0, _, _, _ = commutator_defect(
        model, batch_fn, device, eta=COMM_ETA,
        pad_id_src=pad_id_src, pad_id_tgt=pad_id_tgt)
    d0_cpu = delta0.cpu()
    d0_norm = d0_cpu.norm().item()
    defect_trace.append({
        "step": 0, "defect": D0,
        "attn_frac": d0_cpu[amask].norm().item() / (d0_norm + 1e-15),
        "mlp_frac": d0_cpu[mmask].norm().item() / (d0_norm + 1e-15),
    })

    print(f"  Phase 1: Training with lightweight defect tracking...")
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
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_accum.push()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        if step % cfg.MODEL_LOG_EVERY == 0:
            attn_logs.append({"step": step, "layers": extract_attn_matrices(model)})

        if step % checkpoint_every == 0:
            checkpoints.append(
                (step, {k: v.cpu().clone() for k, v in model.state_dict().items()},
                 {k: (v.cpu().clone() if v is not None else None)
                  for k, v in grad_accum.get_accum().items()})
            )
            D, delta, _, _, _ = commutator_defect(
                model, batch_fn, device, eta=COMM_ETA,
                pad_id_src=pad_id_src, pad_id_tgt=pad_id_tgt)
            d_cpu = delta.cpu()
            d_norm = d_cpu.norm().item()
            defect_trace.append({
                "step": step, "defect": D,
                "attn_frac": d_cpu[amask].norm().item() / (d_norm + 1e-15),
                "mlp_frac": d_cpu[mmask].norm().item() / (d_norm + 1e-15),
            })

        if step % cfg.EVAL_EVERY == 0 or step == 1:
            train_loss, train_acc, train_seq_acc = eval_on_dataset(
                model, train_src, train_tgt_in, train_tgt_out, device,
                pad_id=pad_id_src)
            test_loss, test_acc, test_seq_acc = eval_on_dataset(
                model, test_src, test_tgt_in, test_tgt_out, device,
                pad_id=pad_id_src)

            metrics.append({
                "step": step, "train_loss": train_loss,
                "train_acc": train_acc, "train_seq_acc": train_seq_acc,
                "test_loss": test_loss, "test_acc": test_acc,
                "test_seq_acc": test_seq_acc,
            })

            if test_seq_acc >= cfg.STOP_ACC:
                patience += 1
                if patience >= cfg.STOP_PATIENCE:
                    grokked = True
                    grok_step = step
                    print(f"    GROKKED at step {step} (test_seq_acc={test_seq_acc:.4f})")
                    # Save a post-grok checkpoint if the last one is before grok_step
                    if not checkpoints or checkpoints[-1][0] < step:
                        checkpoints.append(
                            (step, {k: v.cpu().clone() for k, v in model.state_dict().items()},
                             {k: (v.cpu().clone() if v is not None else None)
                              for k, v in grad_accum.get_accum().items()})
                        )
                        D, delta, _, _, _ = commutator_defect(
                            model, batch_fn, device, eta=COMM_ETA,
                            pad_id_src=pad_id_src, pad_id_tgt=pad_id_tgt)
                        d_cpu = delta.cpu()
                        d_norm = d_cpu.norm().item()
                        defect_trace.append({
                            "step": step, "defect": D,
                            "attn_frac": d_cpu[amask].norm().item() / (d_norm + 1e-15),
                            "mlp_frac": d_cpu[mmask].norm().item() / (d_norm + 1e-15),
                        })
                    break
            else:
                patience = 0

        if step % (cfg.EVAL_EVERY * 10) == 0:
            elapsed = (time.time() - t0) / 60
            m = metrics[-1] if metrics else {}
            print(f"    step {step:6d} | "
                  f"train {m.get('train_seq_acc',0):.3f} test {m.get('test_seq_acc',0):.3f} | "
                  f"defect={defect_trace[-1]['defect']:.4f} | {elapsed:.1f}m")

    # -- Phase 2: Select checkpoints --
    print(f"  Phase 2: Selecting 4 phase checkpoints from {len(checkpoints)} saved...")
    selected = _select_phase_checkpoints(checkpoints, metrics, grokked, grok_step)

    for phase, ci in sorted(selected.items(), key=lambda x: x[1]):
        step_at = checkpoints[ci][0]
        closest_m = None
        for m in metrics:
            if m["step"] <= step_at and (closest_m is None or m["step"] > closest_m["step"]):
                closest_m = m
        ta = closest_m["train_seq_acc"] if closest_m else 0
        te = closest_m["test_seq_acc"] if closest_m else 0
        print(f"    {phase:>15s}: ckpt[{ci}] step={step_at}, train={ta:.3f} test={te:.3f}")

    # -- Phase 3: Three-basis measurement --
    print(f"  Phase 3: Three-basis exec/random measurement at 4 phase points...")
    phase_records = []

    for phase_name in PHASE_NAMES:
        ci = selected[phase_name]
        step_at, sd, grad_snap = checkpoints[ci]

        model.load_state_dict(sd)
        model.to(device)

        print(f"    Measuring {phase_name} (step={step_at})...")
        rec = _measure_three_basis(
            model, blocks, total_params, init_weights, grad_snap,
            batch_fn, device, amask, mmask, phase_name, step_at,
            pad_id_src, pad_id_tgt
        )
        phase_records.append(rec)

        w_r = rec["weight_svd_ratio"]
        dw_r = rec["delta_w_svd_ratio"]
        g_r = rec["grad_svd_ratio"]
        print(f"      defect={rec['defect_median']:.4f}  "
              f"W={w_r:.2f}x  DW={dw_r:.2f}x  G={g_r:.2f}x")

    return {
        "phase_records": phase_records,
        "defect_trace": defect_trace,
        "metrics": metrics,
        "grokked": grokked,
        "grok_step": grok_step,
        "attn_logs": attn_logs,
        "block_names": block_names,
        "total_params": total_params,
    }


def _select_phase_checkpoints(checkpoints, metrics, grokked, grok_step):
    selected = {}
    selected["early"] = max(1, len(checkpoints) // 10)

    best_mem_idx, best_mem_gap = None, -1
    for ci, (step, _, _) in enumerate(checkpoints):
        closest_m = None
        for m in metrics:
            if m["step"] <= step and (closest_m is None or m["step"] > closest_m["step"]):
                closest_m = m
        if closest_m and closest_m.get("train_seq_acc", 0) >= 0.90 and closest_m.get("test_seq_acc", 0) < 0.5:
            gap = closest_m["train_seq_acc"] - closest_m["test_seq_acc"]
            if gap > best_mem_gap:
                best_mem_gap = gap
                best_mem_idx = ci
    selected["memorization"] = best_mem_idx if best_mem_idx is not None else len(checkpoints) * 4 // 10

    if grokked and grok_step is not None:
        pre = None
        post = None
        for ci, (step, _, _) in enumerate(checkpoints):
            if step < grok_step:
                pre = ci
            elif post is None and step >= grok_step:
                post = ci
        selected["pre-grok"] = pre if pre is not None else len(checkpoints) - 2
        selected["post-grok"] = post if post is not None else len(checkpoints) - 1
    else:
        selected["pre-grok"] = len(checkpoints) * 8 // 10
        selected["post-grok"] = len(checkpoints) - 1

    for k in selected:
        selected[k] = max(0, min(selected[k], len(checkpoints) - 1))
    return selected


def _measure_three_basis(model, blocks, total_params, init_weights,
                          grad_acc_snapshot, batch_fn, device,
                          amask, mmask, phase_name, step,
                          pad_id_src=0, pad_id_tgt=0):
    model.train()
    deltas_info = []
    for _ in range(COMM_K):
        D_val, delta, gcos, nA, nB = commutator_defect(
            model, batch_fn, device, eta=COMM_ETA,
            pad_id_src=pad_id_src, pad_id_tgt=pad_id_tgt
        )
        deltas_info.append({
            "delta": delta.detach().cpu().float(),
            "nA": nA.cpu().float() if hasattr(nA, 'cpu') else torch.tensor(float(nA)),
            "nB": nB.cpu().float() if hasattr(nB, 'cpu') else torch.tensor(float(nB)),
            "defect": D_val,
        })

    defect_med = float(np.median([d["defect"] for d in deltas_info]))
    med_idx = np.argsort([d["defect"] for d in deltas_info])[len(deltas_info)//2]
    med_delta = deltas_info[med_idx]["delta"]
    d_norm = med_delta.norm().item()
    attn_frac = med_delta[amask].norm().item() / (d_norm + 1e-15)
    mlp_frac  = med_delta[mmask].norm().item() / (d_norm + 1e-15)

    rec = {
        "phase": phase_name, "step": step,
        "defect_median": defect_med,
        "attn_frac": attn_frac, "mlp_frac": mlp_frac,
    }

    basis_fns = {
        "weight_svd": lambda m, b, k=SVD_TOPK: basis_weight_svd(m, b, k),
        "delta_w_svd": lambda m, b, k=SVD_TOPK: basis_delta_w_svd(m, b, init_weights, k),
        "grad_svd": lambda m, b, k=SVD_TOPK: basis_grad_svd(m, b, grad_acc_snapshot, k),
    }

    for btype, bfn in basis_fns.items():
        B = build_joint_basis(model, blocks, total_params, bfn, k=SVD_TOPK)
        K = B.shape[1] if B is not None else 0

        pf_exec_vals, pf_rand_vals = [], []
        for info in deltas_info:
            delta = info["delta"]
            nA, nB = info["nA"], info["nB"]
            pc = projected_commutator(delta, B, nA, nB)
            pf_exec_vals.append(pc["proj"] / (pc["full"] + 1e-15))

            delta_norm = delta.norm().item()
            if K > 0 and delta_norm > 1e-15:
                rand_norms = random_projection_norm(delta, K, n_trials=N_RANDOM_TRIALS)
                pf_rand_vals.append(float(np.mean([rn / (delta_norm + 1e-15)
                                                    for rn in rand_norms])))
            else:
                pf_rand_vals.append(0.0)

        pf_exec = float(np.median(pf_exec_vals))
        pf_rand = float(np.median(pf_rand_vals))
        rec[f"{btype}_pf_exec"] = pf_exec
        rec[f"{btype}_pf_rand"] = pf_rand
        rec[f"{btype}_ratio"] = pf_exec / (pf_rand + 1e-15)
        rec[f"{btype}_K"] = K

    return rec


# =============================================================================
# Main
# =============================================================================

def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    all_results = {}

    for wd in [1.0, 0.0]:
        for seed in SEEDS:
            tag = f"scan_wd{wd}_s{seed}"
            print(f"\n{'='*70}")
            print(f"  {tag}")
            print(f"{'='*70}")

            steps = 20_000 if wd == 0.0 else 30_000
            ckpt_every = CHECKPOINT_EVERY if wd > 0 else 500

            cfg = ScanSweepConfig(WEIGHT_DECAY=wd, SEED=seed, STEPS=steps)
            result = train_and_measure_phases(cfg, checkpoint_every=ckpt_every)
            all_results[(wd, seed)] = result

    # -- Summary table --
    print(f"\n{'='*90}")
    print("  THREE-BASIS EXEC/RANDOM RATIOS AT 4 PHASE POINTS")
    print(f"{'='*90}")

    for (wd, seed), data in sorted(all_results.items()):
        tag = f"wd={wd} s={seed}"
        grok_str = "yes" if data["grokked"] else "no"
        print(f"\n  {tag} (grokked={grok_str}):")
        print(f"    {'phase':>15s}  {'step':>6s}  {'defect':>8s}  "
              f"{'W-SVD':>7s}  {'DW-SVD':>7s}  {'G-SVD':>7s}  "
              f"{'attn%':>6s}  {'mlp%':>6s}")

        for rec in data["phase_records"]:
            print(f"    {rec['phase']:>15s}  {rec['step']:6d}  "
                  f"{rec['defect_median']:8.4f}  "
                  f"{rec['weight_svd_ratio']:6.2f}x  "
                  f"{rec['delta_w_svd_ratio']:6.2f}x  "
                  f"{rec['grad_svd_ratio']:6.2f}x  "
                  f"{rec['attn_frac']:6.1%}  {rec['mlp_frac']:6.1%}")

    # -- Figures --
    print("\n  Generating figures...")
    seed_colors = {42: "#1f77b4", 137: "#ff7f0e", 2024: "#2ca02c"}
    phase_colors = {"early": "#3498db", "memorization": "#e74c3c",
                    "pre-grok": "#f39c12", "post-grok": "#2ecc71"}

    # figJ: Commutator defect over training
    fig, ax = plt.subplots(figsize=(10, 5))
    for seed in SEEDS:
        for wd, ls, alpha in [(1.0, "-", 1.0), (0.0, "--", 0.6)]:
            key = (wd, seed)
            if key not in all_results:
                continue
            dt = all_results[key]["defect_trace"]
            steps = [c["step"] for c in dt]
            defs = [c["defect"] for c in dt]
            label = f"s={seed} wd={wd}"
            ax.plot(steps, defs, label=label, color=seed_colors[seed],
                    linewidth=2 if wd == 1.0 else 1.5, linestyle=ls, alpha=alpha)

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Commutator defect", fontsize=12)
    ax.set_title("SCAN: Commutator Defect During Training", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figJ_scan_commutator_defect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figJ_scan_commutator_defect.png")

    # figN: Attention vs MLP fraction
    fig, axes = plt.subplots(1, len(SEEDS), figsize=(5 * len(SEEDS), 4), squeeze=False)
    for si, seed in enumerate(SEEDS):
        ax = axes[0, si]
        key = (1.0, seed)
        if key not in all_results:
            continue
        dt = all_results[key]["defect_trace"]
        steps = [c["step"] for c in dt]
        attn_fs = [c.get("attn_frac", 0) * 100 for c in dt]
        mlp_fs  = [c.get("mlp_frac", 0) * 100 for c in dt]
        ax.plot(steps, attn_fs, label="Attention", linewidth=2, color="#1f77b4")
        ax.plot(steps, mlp_fs, label="MLP", linewidth=2, color="#9467bd")
        ax.set_title(f"seed={seed}", fontsize=12)
        ax.set_xlabel("Training step")
        ax.set_ylabel("% of ||commutator||")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("SCAN: Attention vs MLP Fraction of Commutator (wd=1.0)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figN_scan_attn_vs_mlp_commutator.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figN_scan_attn_vs_mlp_commutator.png")

    # figO: Three-basis exec/random ratio
    fig, axes = plt.subplots(1, len(SEEDS), figsize=(6 * len(SEEDS), 5.5), squeeze=False)
    for si, seed in enumerate(SEEDS):
        ax = axes[0, si]
        key = (1.0, seed)
        if key not in all_results:
            ax.set_title(f"seed={seed} - no data")
            continue

        data = all_results[key]
        pr = data["phase_records"]

        x = np.arange(len(PHASE_NAMES))
        width = 0.25
        for bi, btype in enumerate(BASIS_TYPES):
            ratios = [r.get(f"{btype}_ratio", 1.0) for r in pr]
            offset = (bi - 1) * width
            bars = ax.bar(x + offset, ratios, width * 0.9,
                         label=BASIS_LABELS[btype],
                         color=BASIS_COLORS[btype], alpha=0.85)
            for bar, ratio in zip(bars, ratios):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"{ratio:.1f}x", ha="center", fontsize=7, fontweight="bold",
                        color=BASIS_COLORS[btype])

        ax.axhline(y=1.0, color="red", linestyle=":", linewidth=2.5,
                   alpha=0.8, label="Random = 1.0")

        labels = [f"{r['phase']}\nstep {r['step']}" for r in pr]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("exec / random ratio")
        ax.set_title(f"seed={seed}" + (f" (grok@{data['grok_step']})"
                     if data["grokked"] else " (no grok)"), fontsize=11)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("SCAN: Three-Basis Exec/Random Ratio at 4 Phase Points (wd=1.0)\n"
                 "ratio > 1 = commutator aligns WITH learning tangent; < 1 = orthogonal",
                 fontsize=12, y=1.06)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figO_scan_three_basis_ratios.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figO_scan_three_basis_ratios.png")

    # figO2: wd comparison
    fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 5), squeeze=False)
    for bi, btype in enumerate(BASIS_TYPES):
        ax = axes[0, bi]
        for wd, color, label_prefix in [(1.0, "#2ca02c", "wd=1.0"), (0.0, "#d62728", "wd=0.0")]:
            phase_ratios = {p: [] for p in PHASE_NAMES}
            for seed in SEEDS:
                key = (wd, seed)
                if key not in all_results:
                    continue
                for rec in all_results[key]["phase_records"]:
                    phase_ratios[rec["phase"]].append(rec.get(f"{btype}_ratio", 1.0))

            phases = PHASE_NAMES
            means = [np.mean(phase_ratios[p]) if phase_ratios[p] else 0 for p in phases]
            stds = [np.std(phase_ratios[p]) if len(phase_ratios[p]) > 1 else 0 for p in phases]

            x = np.arange(len(phases))
            width = 0.35
            offset = -width/2 if wd == 1.0 else width/2
            ax.bar(x + offset, means, width, yerr=stds,
                   label=label_prefix, color=color, alpha=0.85, capsize=3)

        ax.axhline(y=1.0, color="red", linestyle=":", linewidth=2, alpha=0.5)
        ax.set_xticks(np.arange(len(PHASE_NAMES)))
        ax.set_xticklabels(PHASE_NAMES, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("exec / random ratio")
        ax.set_title(BASIS_LABELS[btype], fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("SCAN: Exec/Random Ratio by Phase - wd=1.0 vs wd=0.0\n"
                 "(mean +/- std over 3 seeds)", fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figO2_scan_phase_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figO2_scan_phase_comparison.png")

    # figO3: Exec vs Random breakdown (seed=42)
    fig, ax = plt.subplots(figsize=(10, 5))
    key = (1.0, 42)
    if key in all_results:
        pr = all_results[key]["phase_records"]
        x = np.arange(len(PHASE_NAMES))
        width = 0.25

        for bi, btype in enumerate(BASIS_TYPES):
            exec_pcts = [r.get(f"{btype}_pf_exec", 0) * 100 for r in pr]
            rand_pcts = [r.get(f"{btype}_pf_rand", 0) * 100 for r in pr]
            offset = (bi - 1) * width

            ax.bar(x + offset, exec_pcts, width * 0.45, bottom=0,
                   label=f"{BASIS_LABELS[btype]} exec",
                   color=BASIS_COLORS[btype], alpha=0.85)
            ax.bar(x + offset + width * 0.45, rand_pcts, width * 0.45, bottom=0,
                   label=f"{BASIS_LABELS[btype]} rand",
                   color=BASIS_COLORS[btype], alpha=0.35,
                   edgecolor=BASIS_COLORS[btype], linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['phase']}\nstep {r['step']}" for r in pr], fontsize=9)
        ax.set_ylabel("proj / ||commutator|| (%)")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("Exec vs Random Projection (seed=42, wd=1.0)\n"
                      "Solid = exec basis, transparent = random baseline", fontsize=12)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figO3_scan_exec_vs_random.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figO3_scan_exec_vs_random.png")

    # Save results
    save_path = OUT_DIR / "scan_commutator_results.pt"
    torch.save(all_results, save_path)
    print(f"\n  saved {save_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
