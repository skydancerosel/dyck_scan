#!/usr/bin/env python3
"""
Commutator defect analysis on the Dyck depth-prediction manifold.

Adapts the commutator framework from the grokking codebase to the
Dyck-1 balanced-parentheses depth-prediction task.

Three-basis exec/random ratio comparison (mirroring bubble/grok_multibasis_controls.py):
  1. Weight SVD    — top-k SVD of current weight W
  2. ΔW SVD        — top-k SVD of (W_current - W_init)
  3. Gradient SVD   — top-k SVD of accumulated recent gradients

For each (wd, seed) condition:
  1. Train model, accumulating gradients and saving checkpoints
  2. At each checkpoint, compute commutator defect (K=COMM_K median)
  3. For each of 3 basis types, build joint basis and project commutator
  4. Compare exec projection vs random subspace projection → ratio

Produces:
  figJ  -- Commutator defect over training (wd=1.0 vs wd=0.0)
  figK  -- Projected vs residual commutator (integrability, PCA traj basis)
  figM  -- Defect x integrability combined
  figN  -- Attention weight fraction of commutator
  figO  -- Three-basis exec/random ratio over training (hero figure)
  figO2 -- Bar chart at peak defect: real vs random for all 3 bases
  figO3 -- Per-seed three-basis time series
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
from dyck_grok_sweep import (
    DyckSweepConfig, DyckTransformerLM, make_batch, masked_ce_loss,
    VOCAB_SIZE, get_device, extract_attn_matrices, eval_loss_acc,
    build_depth_dataset, split_dataset, sample_batch,
    eval_on_dataset, flatten_model_params,
)
from dyck_pca_analysis import pca_on_trajectory, collect_trajectory

# -- config -------------------------------------------------------------------
OUT_DIR = Path(__file__).parent / "dyck_pca_plots"
ATTN_KEYS = ["WQ", "WK", "WV", "WO"]
MLP_KEYS  = ["W_up", "W_down"]
WEIGHT_KEYS = ATTN_KEYS + MLP_KEYS
SEEDS = [42, 137, 2024]
CHECKPOINT_EVERY = 200
COMM_K = 9
COMM_ETA = 1e-3
N_PCA_COMPONENTS = 2
SVD_TOPK = 3            # top-k singular directions per weight block
N_RANDOM_TRIALS = 5      # random bases per projection (matches bubble)
GRAD_WINDOW = 50         # accumulate gradients over this many recent steps

BASIS_TYPES = ["weight_svd", "delta_w_svd", "grad_svd"]
BASIS_COLORS = {
    "weight_svd": "#2ecc71",    # Green
    "delta_w_svd": "#3498db",   # Blue
    "grad_svd": "#9b59b6",      # Purple
}
BASIS_LABELS = {
    "weight_svd": "Weight SVD",
    "delta_w_svd": "ΔW SVD",
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
    return torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])


def commutator_defect(model, batch_fn, device, eta=1e-3, eps=1e-12,
                      adaptive=True, min_perturb_norm=1e-6):
    """Scale-normalized commutator defect for Dyck depth prediction.

    If adaptive=True and the gradient perturbation is too small (below
    min_perturb_norm), eta is scaled up so that the step is large enough
    for float32 arithmetic to resolve the commutator.
    """
    was_training = model.training
    model.train()

    def flat_grad(x, y):
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = masked_ce_loss(logits, y)
        loss.backward()
        return _flatten_grad(model)

    xA, yA = batch_fn()
    xB, yB = batch_fn()

    theta0 = _flatten_params(model)
    gA = flat_grad(xA, yA)
    gB = flat_grad(xB, yB)

    gA_norm = gA.norm()
    gB_norm = gB.norm()
    grad_cos = (gA @ gB) / (gA_norm * gB_norm + eps)

    # Adaptive eta: scale up if perturbation too small for float32
    effective_eta = eta
    if adaptive and gA_norm > 0 and gB_norm > 0:
        perturb_norm = min(eta * gA_norm.item(), eta * gB_norm.item())
        if perturb_norm < min_perturb_norm:
            effective_eta = min_perturb_norm / min(gA_norm.item(), gB_norm.item())
            effective_eta = min(effective_eta, 1.0)  # cap at 1.0

    # Path AB
    _write_params(model, theta0 - effective_eta * gA)
    gB1 = flat_grad(xB, yB)
    thetaAB = theta0 - effective_eta * gA - effective_eta * gB1

    # Path BA
    _write_params(model, theta0 - effective_eta * gB)
    gA1 = flat_grad(xA, yA)
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
    """Project delta onto K random orthonormal directions. Returns list of norms."""
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
# Block registry: enumerate all weight blocks with offsets
# =============================================================================

def get_block_registry(model):
    """Returns list of dicts describing each weight block:
      name, layer, offset, shape, numel, row_start
    """
    offsets, total_params = _param_offsets(model)
    blocks = []

    for li, layer in enumerate(model.encoder.layers):
        attn = layer.self_attn
        d = attn.embed_dim

        # WQ, WK, WV from in_proj_weight [3d, d]
        ip_id = id(attn.in_proj_weight)
        ip_off = offsets.get(ip_id, None)
        if ip_off is not None:
            for name, row_start in [("WQ", 0), ("WK", d), ("WV", 2*d)]:
                blocks.append({
                    "name": f"L{li}_{name}",
                    "layer": li,
                    "offset": ip_off + row_start * d,
                    "shape": (d, d),
                    "numel": d * d,
                    "row_start": row_start,
                })

        # WO
        out_id = id(attn.out_proj.weight)
        out_off = offsets.get(out_id, None)
        if out_off is not None:
            blocks.append({
                "name": f"L{li}_WO",
                "layer": li,
                "offset": out_off,
                "shape": attn.out_proj.weight.shape,
                "numel": attn.out_proj.weight.numel(),
                "row_start": None,
            })

        # MLP1 (linear1 = W_up)
        if hasattr(layer, 'linear1'):
            l1_id = id(layer.linear1.weight)
            l1_off = offsets.get(l1_id, None)
            if l1_off is not None:
                blocks.append({
                    "name": f"L{li}_MLP1",
                    "layer": li,
                    "offset": l1_off,
                    "shape": layer.linear1.weight.shape,
                    "numel": layer.linear1.weight.numel(),
                    "row_start": None,
                })

        # MLP2 (linear2 = W_down)
        if hasattr(layer, 'linear2'):
            l2_id = id(layer.linear2.weight)
            l2_off = offsets.get(l2_id, None)
            if l2_off is not None:
                blocks.append({
                    "name": f"L{li}_MLP2",
                    "layer": li,
                    "offset": l2_off,
                    "shape": layer.linear2.weight.shape,
                    "numel": layer.linear2.weight.numel(),
                    "row_start": None,
                })

    return blocks, total_params


def get_block_weight(model, block_info):
    """Extract current weight matrix for a block."""
    li = block_info["layer"]
    layer = model.encoder.layers[li]
    name = block_info["name"]

    if "WQ" in name or "WK" in name or "WV" in name:
        d = layer.self_attn.embed_dim
        ip_w = layer.self_attn.in_proj_weight.detach()
        row_start = block_info["row_start"]
        return ip_w[row_start:row_start+d, :]
    elif "WO" in name:
        return layer.self_attn.out_proj.weight.detach()
    elif "MLP1" in name:
        return layer.linear1.weight.detach()
    elif "MLP2" in name:
        return layer.linear2.weight.detach()
    return None


# =============================================================================
# Three basis constructors (per-block)
# =============================================================================

def _block_basis(block, topk):
    """Top-k singular directions (flattened) for a weight block."""
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
    """Basis 1: top-k SVD of current weight W."""
    W = get_block_weight(model, block_info)
    if W is None:
        return []
    return _block_basis(W, k)


def basis_delta_w_svd(model, block_info, init_weights, k=3):
    """Basis 2: top-k SVD of (W_current - W_init) = weight displacement."""
    W = get_block_weight(model, block_info)
    if W is None:
        return []
    W0 = init_weights[block_info["name"]]
    delta_W = W.cpu() - W0.cpu()
    if delta_W.norm() < 1e-10:
        return _block_basis(W, k)  # fallback to weight SVD
    return _block_basis(delta_W, k)


def basis_grad_svd(model, block_info, grad_accum, k=3):
    """Basis 3: top-k SVD of accumulated gradient matrix."""
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
# Joint basis construction + gradient accumulator
# =============================================================================

def build_joint_basis(model, blocks, total_params, basis_fn, k=3):
    """Build joint basis [P, K] from per-block bases embedded in full space."""
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
        """Capture current .grad for each param, store per-block."""
        grads = {}
        for b in self.blocks:
            li = b["layer"]
            layer = self.model.encoder.layers[li]
            name = b["name"]

            if "WQ" in name or "WK" in name or "WV" in name:
                p = layer.self_attn.in_proj_weight
                if p.grad is not None:
                    d = layer.self_attn.embed_dim
                    row_start = b["row_start"]
                    grads[name] = p.grad[row_start:row_start+d, :].detach().cpu().clone()
                else:
                    grads[name] = None
            elif "WO" in name:
                p = layer.self_attn.out_proj.weight
                grads[name] = p.grad.detach().cpu().clone() if p.grad is not None else None
            elif "MLP1" in name:
                p = layer.linear1.weight
                grads[name] = p.grad.detach().cpu().clone() if p.grad is not None else None
            elif "MLP2" in name:
                p = layer.linear2.weight
                grads[name] = p.grad.detach().cpu().clone() if p.grad is not None else None

        self.buffer.append(grads)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)

    def get_accum(self):
        """Return accumulated gradient per block (sum over window)."""
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
# PCA trajectory basis (kept for backward compatibility with figK/figM)
# =============================================================================

def build_pca_basis(model, attn_logs, n_components=2, device="cpu",
                    include_mlp=True):
    """Build orthonormal basis from PCA directions embedded in full param space."""
    offsets, total_params = _param_offsets(model)
    basis_vecs = []

    for layer_idx, layer in enumerate(model.encoder.layers):
        attn_mod = layer.self_attn
        d = attn_mod.embed_dim

        in_proj_id = id(attn_mod.in_proj_weight)
        out_proj_id = id(attn_mod.out_proj.weight)
        if in_proj_id not in offsets or out_proj_id not in offsets:
            continue

        in_proj_offset = offsets[in_proj_id]
        out_proj_offset = offsets[out_proj_id]

        for wkey, local_start in [("WQ", 0), ("WK", d*d), ("WV", 2*d*d)]:
            _, mats = collect_trajectory(attn_logs, layer_idx, wkey)
            if len(mats) == 0:
                continue
            pca = pca_on_trajectory(mats, top_k=n_components)
            if pca is None:
                continue
            n_avail = min(n_components, len(pca["components"]))
            for ci in range(n_avail):
                direction = torch.from_numpy(pca["components"][ci]).float()
                gv = torch.zeros(total_params, device=device)
                start = in_proj_offset + local_start
                gv[start:start + d*d] = direction.to(device)
                basis_vecs.append(gv)

        _, mats = collect_trajectory(attn_logs, layer_idx, "WO")
        if len(mats) == 0:
            continue
        pca = pca_on_trajectory(mats, top_k=n_components)
        if pca is None:
            continue
        n_avail = min(n_components, len(pca["components"]))
        for ci in range(n_avail):
            direction = torch.from_numpy(pca["components"][ci]).float()
            gv = torch.zeros(total_params, device=device)
            gv[out_proj_offset:out_proj_offset + d*d] = direction.to(device)
            basis_vecs.append(gv)

        if include_mlp:
            for mlp_key, attr_name in [("W_up", "linear1"), ("W_down", "linear2")]:
                if not hasattr(layer, attr_name):
                    continue
                mod = getattr(layer, attr_name)
                w_id = id(mod.weight)
                if w_id not in offsets:
                    continue
                w_offset = offsets[w_id]
                n_elem = mod.weight.numel()

                _, mats = collect_trajectory(attn_logs, layer_idx, mlp_key)
                if len(mats) == 0:
                    continue
                pca = pca_on_trajectory(mats, top_k=n_components)
                if pca is None:
                    continue
                n_avail = min(n_components, len(pca["components"]))
                for ci in range(n_avail):
                    direction = torch.from_numpy(pca["components"][ci]).float()
                    gv = torch.zeros(total_params, device=device)
                    gv[w_offset:w_offset + n_elem] = direction.to(device)
                    basis_vecs.append(gv)

    if not basis_vecs:
        return None
    B = torch.stack(basis_vecs, dim=1)
    B_ortho, _ = torch.linalg.qr(B.cpu(), mode="reduced")
    return B_ortho.to(device)


# =============================================================================
# Masks for attn/mlp fraction
# =============================================================================

def attn_weight_mask(model):
    offsets, total_params = _param_offsets(model)
    mask = torch.zeros(total_params, dtype=torch.bool)
    for layer in model.encoder.layers:
        attn_mod = layer.self_attn
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
    for layer in model.encoder.layers:
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
# Training with lightweight defect tracking + checkpoint saving at phase points
# =============================================================================

PHASE_NAMES = ["early", "memorization", "pre-grok", "post-grok"]

def train_and_measure_phases(cfg, checkpoint_every=200):
    """Train model, identify 4 phase points, do three-basis measurement there.

    Phase 1: Train fully with lightweight defect tracking, saving state_dicts
             at every checkpoint_every steps.
    Phase 2: From metrics, identify 4 representative checkpoints.
    Phase 3: Reload each checkpoint, do expensive three-basis measurement.
    """
    device = get_device()

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    # Deterministic dataset from DATA_SEED
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
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    )

    blocks, total_params = get_block_registry(model)
    block_names = [b["name"] for b in blocks]
    print(f"    P={total_params}, {len(blocks)} blocks: {block_names}")

    init_weights = {}
    for b in blocks:
        init_weights[b["name"]] = get_block_weight(model, b).cpu().clone()

    grad_accum = GradAccumulator(blocks, model, window=GRAD_WINDOW)
    amask = attn_weight_mask(model)
    mmask = mlp_weight_mask(model)

    def batch_fn():
        return sample_batch(train_x, train_y, cfg.BATCH_SIZE, device)

    attn_logs = [{"step": 0, "layers": extract_attn_matrices(model)}]
    # Save (step, state_dict, grad_accum_snapshot) — keep all for phase selection
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
    D0, delta0, _, _, _ = commutator_defect(model, batch_fn, device, eta=COMM_ETA)
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
        x, y = sample_batch(train_x, train_y, cfg.BATCH_SIZE, device)
        logits = model(x)
        loss = masked_ce_loss(logits, y)
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
            D, delta, _, _, _ = commutator_defect(model, batch_fn, device, eta=COMM_ETA)
            d_cpu = delta.cpu()
            d_norm = d_cpu.norm().item()
            defect_trace.append({
                "step": step, "defect": D,
                "attn_frac": d_cpu[amask].norm().item() / (d_norm + 1e-15),
                "mlp_frac": d_cpu[mmask].norm().item() / (d_norm + 1e-15),
            })

        if step % cfg.EVAL_EVERY == 0 or step == 1:
            train_loss, train_acc = eval_on_dataset(model, train_x, train_y, device)
            test_loss, test_acc = eval_on_dataset(model, test_x, test_y, device)

            ood_rng = random.Random(cfg.SEED + 777 + step)
            ood_loss, ood_acc = eval_loss_acc(
                model, cfg.EVAL_BATCHES_OOD, cfg.EVAL_BS_OOD,
                cfg.CTX_LEN_OOD, cfg.MAX_PAIRS_OOD, device, ood_rng)

            metrics.append({
                "step": step, "train_loss": train_loss, "train_acc": train_acc,
                "test_loss": test_loss, "test_acc": test_acc,
                "ood_loss": ood_loss, "ood_acc": ood_acc,
            })

            if test_acc >= cfg.STOP_ACC:
                patience += 1
                if patience >= cfg.STOP_PATIENCE:
                    grokked = True
                    grok_step = step
                    print(f"    GROKKED at step {step} (test_acc={test_acc:.4f})")
                    # Save a post-grok checkpoint if the last one is before grok_step
                    if not checkpoints or checkpoints[-1][0] < step:
                        checkpoints.append(
                            (step, {k: v.cpu().clone() for k, v in model.state_dict().items()},
                             {k: (v.cpu().clone() if v is not None else None)
                              for k, v in grad_accum.get_accum().items()})
                        )
                        D, delta, _, _, _ = commutator_defect(model, batch_fn, device, eta=COMM_ETA)
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
                  f"train {m.get('train_acc',0):.3f} test {m.get('test_acc',0):.3f} | "
                  f"defect={defect_trace[-1]['defect']:.4f} | {elapsed:.1f}m")

    # ── Phase 2: Select 4 representative checkpoints ──────────────────
    print(f"  Phase 2: Selecting 4 phase checkpoints from {len(checkpoints)} saved...")
    selected = _select_phase_checkpoints(checkpoints, metrics, grokked, grok_step)

    for phase, ci in sorted(selected.items(), key=lambda x: x[1]):
        step_at = checkpoints[ci][0]
        # find closest metric
        closest_m = None
        for m in metrics:
            if m["step"] <= step_at and (closest_m is None or m["step"] > closest_m["step"]):
                closest_m = m
        ta = closest_m["train_acc"] if closest_m else 0
        te = closest_m["test_acc"] if closest_m else 0
        print(f"    {phase:>15s}: ckpt[{ci}] step={step_at}, train={ta:.3f} test={te:.3f}")

    # ── Phase 3: Expensive three-basis measurement at 4 points ────────
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
            batch_fn, device, amask, mmask, phase_name, step_at
        )
        phase_records.append(rec)

        w_r = rec["weight_svd_ratio"]
        dw_r = rec["delta_w_svd_ratio"]
        g_r = rec["grad_svd_ratio"]
        print(f"      defect={rec['defect_median']:.4f}  "
              f"W={w_r:.2f}x  ΔW={dw_r:.2f}x  G={g_r:.2f}x")

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
    """Pick 4 representative checkpoints from saved list."""
    selected = {}

    # Early: ~10% into training
    selected["early"] = max(1, len(checkpoints) // 10)

    # Memorization: max(train_acc - test_acc) where train_acc > 0.90
    best_mem_idx, best_mem_gap = None, -1
    for ci, (step, _, _) in enumerate(checkpoints):
        closest_m = None
        for m in metrics:
            if m["step"] <= step and (closest_m is None or m["step"] > closest_m["step"]):
                closest_m = m
        if closest_m and closest_m["train_acc"] >= 0.90 and closest_m["test_acc"] < 0.5:
            gap = closest_m["train_acc"] - closest_m["test_acc"]
            if gap > best_mem_gap:
                best_mem_gap = gap
                best_mem_idx = ci
    selected["memorization"] = best_mem_idx if best_mem_idx is not None else len(checkpoints) * 4 // 10

    if grokked and grok_step is not None:
        # Pre-grok: last checkpoint before grok_step
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
                          amask, mmask, phase_name, step):
    """Full three-basis exec/random measurement at one checkpoint."""
    model.train()

    deltas_info = []
    for _ in range(COMM_K):
        D_val, delta, gcos, nA, nB = commutator_defect(
            model, batch_fn, device, eta=COMM_ETA
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
            tag = f"dyck_wd{wd}_s{seed}"
            print(f"\n{'='*70}")
            print(f"  {tag}")
            print(f"{'='*70}")

            steps = 10_000 if wd == 0.0 else 20_000
            ckpt_every = CHECKPOINT_EVERY if wd > 0 else 500

            cfg = DyckSweepConfig(WEIGHT_DECAY=wd, SEED=seed, STEPS=steps)

            result = train_and_measure_phases(cfg, checkpoint_every=ckpt_every)
            all_results[(wd, seed)] = result

    # -- Summary table ----------------------------------------------------------
    print(f"\n{'='*90}")
    print("  THREE-BASIS EXEC/RANDOM RATIOS AT 4 PHASE POINTS")
    print(f"{'='*90}")

    for (wd, seed), data in sorted(all_results.items()):
        tag = f"wd={wd} s={seed}"
        grok_str = "yes" if data["grokked"] else "no"
        print(f"\n  {tag} (grokked={grok_str}):")
        print(f"    {'phase':>15s}  {'step':>6s}  {'defect':>8s}  "
              f"{'W-SVD':>7s}  {'ΔW-SVD':>7s}  {'G-SVD':>7s}  "
              f"{'attn%':>6s}  {'mlp%':>6s}")

        for rec in data["phase_records"]:
            print(f"    {rec['phase']:>15s}  {rec['step']:6d}  "
                  f"{rec['defect_median']:8.4f}  "
                  f"{rec['weight_svd_ratio']:6.2f}x  "
                  f"{rec['delta_w_svd_ratio']:6.2f}x  "
                  f"{rec['grad_svd_ratio']:6.2f}x  "
                  f"{rec['attn_frac']:6.1%}  {rec['mlp_frac']:6.1%}")

    # -- Figures ----------------------------------------------------------------
    print("\n  Generating figures...")
    seed_colors = {42: "#1f77b4", 137: "#ff7f0e", 2024: "#2ca02c"}
    phase_colors = {"early": "#3498db", "memorization": "#e74c3c",
                    "pre-grok": "#f39c12", "post-grok": "#2ecc71"}

    # figJ: Commutator defect over training (from lightweight trace)
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
    ax.set_title("Dyck Depth: Commutator Defect During Training", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figJ_dyck_commutator_defect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figJ_dyck_commutator_defect.png")

    # figN: Attention vs MLP fraction over training
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
    fig.suptitle("Dyck Depth: Attention vs MLP Fraction of Commutator (wd=1.0)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figN_dyck_attn_vs_mlp_commutator.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figN_dyck_attn_vs_mlp_commutator.png")

    # =========================================================================
    # figO: THREE-BASIS EXEC/RANDOM RATIO — bar chart by phase (hero figure)
    # =========================================================================
    fig, axes = plt.subplots(1, len(SEEDS), figsize=(6 * len(SEEDS), 5.5), squeeze=False)
    for si, seed in enumerate(SEEDS):
        ax = axes[0, si]
        key = (1.0, seed)
        if key not in all_results:
            ax.set_title(f"seed={seed} — no data")
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
            # Annotate
            for bar, ratio in zip(bars, ratios):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"{ratio:.1f}×", ha="center", fontsize=7, fontweight="bold",
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

    fig.suptitle("Dyck Depth: Three-Basis Exec/Random Ratio at 4 Phase Points (wd=1.0)\n"
                 "ratio > 1 = commutator aligns WITH learning tangent; < 1 = orthogonal",
                 fontsize=12, y=1.06)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figO_dyck_three_basis_ratios.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figO_dyck_three_basis_ratios.png")

    # =========================================================================
    # figO2: Grouped bar — wd=1.0 vs wd=0.0 at each phase
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 5), squeeze=False)
    for bi, btype in enumerate(BASIS_TYPES):
        ax = axes[0, bi]
        for wd, color, label_prefix in [(1.0, "#2ca02c", "wd=1.0"), (0.0, "#d62728", "wd=0.0")]:
            # Average ratio across seeds for each phase
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

    fig.suptitle("Dyck Depth: Exec/Random Ratio by Phase — wd=1.0 vs wd=0.0\n"
                 "(mean ± std over 3 seeds)", fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figO2_dyck_phase_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figO2_dyck_phase_comparison.png")

    # =========================================================================
    # figO3: Exec vs Random breakdown (stacked per phase, seed=42)
    # =========================================================================
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
    fig.savefig(OUT_DIR / "figO3_dyck_exec_vs_random.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figO3_dyck_exec_vs_random.png")

    # -- Save raw results -------------------------------------------------------
    save_path = OUT_DIR / "dyck_commutator_results.pt"
    torch.save(all_results, save_path)
    print(f"\n  saved {save_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
