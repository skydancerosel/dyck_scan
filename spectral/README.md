# Spectral Edge Analysis Scripts

This directory contains the analysis scripts for two papers:

1. **Thesis Table 7 replication** — Gram matrix spectral edge analysis for the original spectral edge thesis.
2. **The Lifecycle of the Spectral Edge** — Companion paper decomposing the edge into gradient and weight-decay phases.

---

## Paper 1: Spectral Edge Thesis Replication

**Script**: `thesis_table7_replication.py`

Computes the three Gram matrix diagnostics (`g_{23}`, gap ratio `R`, weighted `k*`) from rolling-window parameter updates for Dyck-1 and SCAN checkpoints in `dyck_sweep_results/` and `scan_sweep_results/`.

See `../SPECTRAL_EDGE_THESIS_REPLICATION.md` for results.

---

## Paper 2: The Lifecycle of the Spectral Edge

> Xu, Y. (2026). *The Lifecycle of the Spectral Edge: From Gradient Learning to Weight-Decay Compression*. arXiv:2604.07380.

The paper decomposes the dominant Gram singular vector `v₁` into a two-phase lifecycle (learning → compression) and establishes causal properties through six experiments.

### Pipeline

Scripts are listed in the order they should be run. Each produces figures and `.pt` result files under `fourier_dyck_plots/` and `fourier_scan_plots/` (both gitignored).

#### Step 0: Retraining with full state_dict logging

Needed because the original sweep checkpoints only stored attention weight snapshots. For perturbation and probing experiments we need the full model state.

```bash
python3 retrain_dyck_fourier.py   # ~2 min
python3 retrain_scan_fourier.py   # ~20 min (CPU); saves two ~500 MB checkpoints
```

Produces: `fourier_{dyck,scan}_checkpoints/*.pt`

#### Step 1: Fourier structure of the learned models (§8 of the paper)

```bash
python3 fourier_functional_dyck.py         # positional DFT of hidden reps
python3 fourier_functional_scan.py
python3 dyck_depth_basis.py                # depth-conditioned representation geometry
python3 dyck_intermediate_probing.py       # layer-wise linear probes
python3 scan_intermediate_probing.py
python3 dyck_composition_test.py           # token × cumsum factorization
python3 scan_composition_test.py           # verb × direction × repetition
python3 dyck_attention_modes.py            # attention entropy, counting score
python3 scan_attention_modes.py            # encoder-self / decoder-self / cross attention
```

#### Step 2: Gram edge functional modes (§3 of the paper)

```bash
python3 gram_edge_functional_modes.py          # perturbation response f_k(x)
python3 gram_edge_functional_modes_scan.py
python3 principal_direction_analysis.py        # W_Q / W_K SVD direction content
python3 fourier_correct_basis.py               # Fourier in depth basis + per-block decomposition
```

#### Step 3: The mechanism — grad-WD decomposition (§3 of the paper)

```bash
python3 grad_vs_wd_decomposition.py            # decompose Δθ into gradient and WD components per direction
python3 perturbation_curves.py                 # ε-sweep along edge vs bulk directions
python3 edge_rotation_stability.py             # subspace angles over time (both tasks)
```

#### Step 4: Ablation paradox and causal tests (§4–§6 of the paper)

```bash
python3 edge_ablation.py                       # ablation + Hessian curvature + path-norm
python3 random_direction_controls.py           # random direction ablation controls
python3 nonlinear_probes.py                    # linear / quadratic / MLP probes
python3 wd_intervention.py                     # remove/vary weight decay post-grok
python3 loss_decomposition.py                  # α_j · G_j^train · G_j^val (thesis Test 8)
```

#### Step 5: Replication and controls (§9 of the paper)

```bash
python3 multiseed_replication.py               # seeds 42, 137, 2024
python3 scan_full_suite.py                     # SCAN versions of all causal experiments
```

#### Step 6: Paper figures

```bash
python3 paper_figure_dyck_fourier.py           # multi-panel Fourier analysis (Dyck)
python3 paper_figure_scan_fourier.py           # multi-panel Fourier analysis (SCAN)
python3 paper_figure_thesis_connection.py      # 3-panel WD / grad-WD / Hessian figure (paper's Fig. 1)
```

---

## Script-to-paper-section mapping

| Paper Section | Scripts |
|---------------|---------|
| §2 Setup | `retrain_{dyck,scan}_fourier.py` |
| §3 Mechanism (grad-WD) | `grad_vs_wd_decomposition.py`, `paper_figure_thesis_connection.py` |
| §3.4 W_Q / W_K | `principal_direction_analysis.py` |
| §4 Ablation paradox | `edge_ablation.py`, `random_direction_controls.py`, `perturbation_curves.py` |
| §5 Nonlinear re-encoding | `nonlinear_probes.py` |
| §6 WD intervention | `wd_intervention.py` |
| §7 Universality classes | synthesis across Dyck / SCAN results |
| §8 Fourier structure | `fourier_functional_{dyck,scan}.py`, `dyck_{depth_basis,attention_modes,composition_test}.py`, `fourier_correct_basis.py` |
| §9 Controls | `multiseed_replication.py`, `scan_full_suite.py`, `edge_rotation_stability.py`, `loss_decomposition.py` |

---

## Key results

### §3 — The grad-WD flip on `v_1`

| Phase | Dyck grad% | SCAN grad% |
|-------|------------|------------|
| Pre-grok | 97.6% | 88.7% |
| At grok | **5.3%** | **0.2%** |
| Post-grok | 99.5% | 2.1% |

### §4 — Ablation paradox

| Phase | Edge ablation Δacc | Random ablation Δacc |
|-------|-------------------|---------------------|
| At grok | −0.26 ± 0.05 | +0.000 ± 0.000 |
| Late | **−0.58 ± 0.09** | +0.000 ± 0.000 |

Edge is >4000× more impactful than random directions of the same dimensionality.

### §5 — Probe R²

| Phase | Linear | MLP |
|-------|--------|-----|
| At grok | 0.982 ± 0.003 | 0.992 ± 0.002 |
| Late | 0.862 ± 0.011 | **0.990 ± 0.003** |

### §6 — WD intervention

| Condition | Accuracy | Linear R² | Entropy | ‖θ‖ |
|-----------|----------|-----------|---------|-----|
| ω = 0 | 0.973 | **0.987** | 2.15 | 40.4 |
| ω = 1 | 0.982 | 0.851 | 2.27 | 15.3 |
| ω = 2 | **0.985** | 0.709 | **2.28** | **14.3** |
