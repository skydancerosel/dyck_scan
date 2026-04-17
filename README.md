# Spectral Geometry of Grokking on SCAN and Dyck

Code for three papers studying grokking on two sequence-learning tasks:

- **SCAN** compositional generalization (encoder-decoder transformer, ~1.5M params)
- **Dyck-1** depth prediction (causal decoder transformer, ~150k params)

## Papers

1. **Early-Warning Signals of Grokking via Loss-Landscape Geometry** ([arXiv:2602.16967](https://arxiv.org/abs/2602.16967))
   The commutator defect of successive gradient updates rises before grokking with superlinear scaling. Extends the framework from modular arithmetic to structurally diverse sequence tasks.

2. **The Spectral Edge Thesis** ([arXiv:2603.28964](https://arxiv.org/abs/2603.28964)) — replicated on Dyck/SCAN
   Gram matrix eigenvalue gap g₂₃ compresses 33× (Dyck) and 43× (SCAN) during grokking. 24/24 grok runs show compression; 0/24 controls do.

3. **The Lifecycle of the Spectral Edge: From Gradient Learning to Weight-Decay Compression** ([arXiv:2604.07380](https://arxiv.org/pdf/2604.07380))
   Decomposes the dominant Gram direction v₁ into a **two-phase lifecycle**: gradient-driven learning → weight-decay compression. The transition happens at grokking, when gradient and weight decay align along v₁.

## Key Results

### Commutator Defect as Early-Warning Signal (Paper 1)

The commutator defect --- measuring non-commutativity of successive gradient updates --- reliably rises before grokking. Lead time follows a **superlinear power law**:

| Task | Exponent (α) | R²   | Data points |
|------|:------------:|:-----:|:-----------:|
| SCAN | 1.18         | 0.990 | 11          |
| Dyck | 1.13         | 0.908 | 14          |

All three task families (adding mod-arith) exhibit α > 1, yielding advance warning windows of 90-97% at slow learning rates.

### Spectral Edge Lifecycle (Paper 3)

The dominant Gram singular vector v₁ undergoes a sharp transition at grokking:

| Phase | Dyck grad% on v₁ | SCAN grad% on v₁ |
|-------|-----------------|------------------|
| Pre-grok | 97.6% | 88.7% |
| **At grok** | **5.3%** | **0.2%** |
| Post-grok | 99.5% | 2.1% |

Causal properties (Dyck, 3 seeds):

- **Ablation**: removing v₁+v₂ drops accuracy by −0.58 ± 0.09 (>4000× more impactful than random directions)
- **Perturbation flatness**: perturbing along v₁ gives max KL ≈ 0.00005 (Hessian curvature 0.08)
- **Nonlinear re-encoding**: linear probe R² drops 0.98 → 0.86 while MLP probe stays at 0.990 ± 0.003
- **WD intervention**: removing weight decay post-grok recovers linear R² to 0.987 while preserving accuracy at 0.973

Three universality classes emerge:

| Class | Task | Late grad% | Rotation | Func R² |
|-------|------|-----------|----------|---------|
| Functional | Mod-arith | high | moderate | high |
| **Mixed** | **Dyck** | 87% | 18° | 0.80 |
| **Compression** | **SCAN** | 2% | 8° (frozen) | 0.04 (empty) |

## Repository Structure

```
scan/                           # SCAN experiments (Paper 1)
  grok_sweep.py                   # Grokking sweep across learning rates
  lr_sweep.py                     # LR sweep with commutator tracking
  lr_sweep_extended.py            # Extended sweep (multi-seed)
  commutator_analysis.py          # Commutator defect analysis
  generalization_dynamics.py      # Generalization dynamics tracking
  intervention.py                 # Causal intervention experiments
  pca_analysis.py                 # Weight-space PCA analysis
  pc1_lr_experiment.py            # PC1 trajectory experiments
  spectral_geometry.py            # Spectral geometry of attention weights
  figures/                        # Paper 1 figures

dyck/                           # Dyck experiments (Paper 1)
  grok_sweep.py                   # Grokking sweep across learning rates
  lr_sweep.py                     # LR sweep with commutator tracking
  lr_sweep_extended.py            # Extended sweep (multi-seed)
  commutator_analysis.py          # Commutator defect analysis
  generalization_dynamics.py      # Generalization dynamics tracking
  intervention.py                 # Causal intervention experiments
  pca_analysis.py                 # Weight-space PCA analysis
  spectral_geometry.py            # Spectral geometry of attention weights
  figures/                        # Paper 1 figures

spectral/                       # Papers 2 and 3: spectral edge analysis
  README.md                       # Script-by-script paper mapping (IMPORTANT)
  thesis_table7_replication.py    # Paper 2: Gram matrix g₂₃, R, k*

  # Paper 3 pipeline (in order):
  retrain_{dyck,scan}_fourier.py  # Full state_dict logging for inference
  fourier_functional_{dyck,scan}.py     # Positional DFT of hidden reps
  dyck_depth_basis.py             # Depth representation geometry
  {dyck,scan}_intermediate_probing.py   # Layer-wise linear probes
  {dyck,scan}_composition_test.py       # Compositional structure
  {dyck,scan}_attention_modes.py  # Attention entropy / counting
  gram_edge_functional_modes{,_scan}.py # Perturbation response f_k(x)
  principal_direction_analysis.py # W_Q / W_K SVD direction content
  fourier_correct_basis.py        # Fourier in depth basis
  grad_vs_wd_decomposition.py     # The mechanism — Δθ = grad + WD
  perturbation_curves.py          # ε-sweep along edge vs bulk
  edge_rotation_stability.py      # Subspace rotation over time
  edge_ablation.py                # Ablation + Hessian curvature + path-norm
  random_direction_controls.py    # Random vs edge ablation
  nonlinear_probes.py             # Linear / quadratic / MLP probes
  wd_intervention.py              # Weight decay intervention (causal test)
  loss_decomposition.py           # α_j·G_j^train·G_j^val (thesis Test 8)
  multiseed_replication.py        # Seeds 42, 137, 2024
  scan_full_suite.py              # SCAN versions of all experiments
  paper_figure_*.py               # Multi-panel paper figures

paper/                          # Paper 3 source
  main.tex                        # LaTeX source
  arxiv_submission/               # arXiv upload package

scan_data/                      # SCAN train/test splits
SPECTRAL_GEOMETRY_RESULTS.md    # Detailed Paper 1 results
SPECTRAL_EDGE_THESIS_REPLICATION.md  # Paper 2 results
FOURIER_FUNCTIONAL_DYCK_RESULTS.md   # Paper 3 Dyck findings
FOURIER_FUNCTIONAL_SCAN_RESULTS.md   # Paper 3 SCAN findings
SPECTRAL_EDGE_FUNCTIONAL_MODES_SYNTHESIS.md  # Cross-paper synthesis
SPECTRAL_EDGE_THESIS_CONNECTION.md   # Connection to the thesis framework
```

## Requirements

- Python 3.8+
- PyTorch, NumPy, Matplotlib, SciPy
- scikit-learn (for probe experiments in Paper 3)

## Usage

### Paper 1: Commutator defect (early-warning signal)

```bash
python scan/lr_sweep.py
python dyck/lr_sweep.py
python scan/commutator_analysis.py
python dyck/commutator_analysis.py
python scan/intervention.py
python dyck/intervention.py
```

### Paper 2: Gram matrix spectral edge

```bash
MAX_FILE_MB=2500 python3 -u spectral/thesis_table7_replication.py
```

### Paper 3: The lifecycle of the spectral edge

See `spectral/README.md` for the full pipeline. Short version:

```bash
# Step 0: retrain with full state_dict (~25 min total)
python3 spectral/retrain_dyck_fourier.py
python3 spectral/retrain_scan_fourier.py

# Step 1: the mechanism — grad-WD decomposition
python3 spectral/grad_vs_wd_decomposition.py

# Step 2: ablation paradox
python3 spectral/edge_ablation.py
python3 spectral/random_direction_controls.py
python3 spectral/perturbation_curves.py

# Step 3: nonlinear re-encoding
python3 spectral/nonlinear_probes.py

# Step 4: causal WD intervention
python3 spectral/wd_intervention.py

# Step 5: multi-seed replication
python3 spectral/multiseed_replication.py
python3 spectral/scan_full_suite.py

# Paper figures
python3 spectral/paper_figure_thesis_connection.py  # Paper's Fig. 1
```

## Citation

```bibtex
@article{xu2026earlywarning,
  title={Early-Warning Signals of Grokking via Loss-Landscape Geometry},
  author={Xu, Yongzhong},
  year={2026},
  eprint={2602.16967},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2602.16967}
}

@article{xu2026spectral_edge,
  title={The Spectral Edge Thesis: A Mathematical Framework for Intra-Signal Phase Transitions in Neural Network Training},
  author={Xu, Yongzhong},
  year={2026},
  eprint={2603.28964},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2603.28964}
}

@article{xu2026lifecycle,
  title={The Lifecycle of the Spectral Edge: From Gradient Learning to Weight-Decay Compression},
  author={Xu, Yongzhong},
  year={2026},
  eprint={2604.07380},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2604.07380}
}
```

## License

MIT
