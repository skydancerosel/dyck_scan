# Integrability and Grokking on SCAN and Dyck

We show that loss-landscape curvature, measured via the **commutator defect** of successive gradient updates, provides a reliable early-warning signal for grokking. The defect rises systematically before generalization onset, with lead times following a superlinear power law across three task families and over two orders of magnitude in learning rate.

**Paper**: [Early-Warning Signals of Grokking via Loss-Landscape Geometry](https://arxiv.org/abs/2602.16967)

Code and figures for studying grokking through the lens of loss-landscape geometry on two sequence-learning tasks:

- **SCAN** compositional generalization (encoder-decoder transformer, ~1.5M params)
- **Dyck-1** depth prediction (causal decoder transformer, ~150k params)

This extends the commutator defect framework from modular arithmetic to structurally diverse tasks, and introduces a spectral geometry analysis of attention weight matrices.

## Key Results

### Commutator Defect as Early-Warning Signal

The commutator defect --- measuring non-commutativity of successive gradient updates --- reliably rises before the onset of generalization. The lead time follows a **superlinear power law**:

| Task | Exponent (alpha) | R^2   | Data points |
|------|:-----------------:|:-----:|:-----------:|
| SCAN | 1.18              | 0.990 | 11          |
| Dyck | 1.13              | 0.908 | 14          |

Combined with prior results on modular arithmetic, all three task families exhibit alpha > 1, yielding advance warning windows of 90-97% at slow learning rates.

### Spectral Geometry of Attention Weights

SVD analysis of W_Q and W_K reveals a consistent temporal ordering across all 6 grokking runs (3 seeds x 2 tasks), with 0 false positives across 6 controls:

```
sigma_1 ~ sigma_2 (init) -> D_SGD spike -> ||[W_Q, W_K]|| peak -> grok -> C collapse -> sigma_1 >> sigma_2
```

- **SGD defect spike precedes grokking**: 6/6 (100%), mean lead 800 steps (Dyck), 2033 steps (SCAN)
- **Commutator peak precedes grokking**: 6/6 (100%), mean lead 400 steps (Dyck), 1667 steps (SCAN)
- **Controls never grok**: 0/6 (0%)

Phase portraits in the (spectral gap, non-commutativity) plane show grokking trajectories sweeping from high-commutator/low-gap to low-commutator/high-gap, while controls remain stationary.

## Repository Structure

```
scan/                           # SCAN experiments
  grok_sweep.py                   # Grokking sweep across learning rates
  lr_sweep.py                     # LR sweep with commutator tracking
  lr_sweep_extended.py            # Extended sweep (multi-seed)
  commutator_analysis.py          # Commutator defect analysis and plotting
  generalization_dynamics.py      # Generalization dynamics tracking
  intervention.py                 # Causal intervention experiments
  pca_analysis.py                 # Weight-space PCA analysis
  pc1_lr_experiment.py            # PC1 trajectory experiments
  lm_pilot.py                     # Language model pilot
  spectral_geometry.py            # Spectral geometry of attention weights
  figures/                        # All SCAN figures

dyck/                           # Dyck experiments
  grok_sweep.py                   # Grokking sweep across learning rates
  lr_sweep.py                     # LR sweep with commutator tracking
  lr_sweep_extended.py            # Extended sweep (multi-seed)
  lr1e2_diagnostic.py             # Diagnostic for lr=1e-2 instability
  commutator_analysis.py          # Commutator defect analysis and plotting
  generalization_dynamics.py      # Generalization dynamics tracking
  intervention.py                 # Causal intervention experiments
  pca_analysis.py                 # Weight-space PCA analysis
  pc1_lr_experiment.py            # PC1 trajectory experiments
  lm_pilot.py                     # Language model pilot
  spectral_geometry.py            # Spectral geometry of attention weights
  figures/                        # All Dyck figures

grok_pc1_lr_experiment.py       # Cross-task joint PC1 analysis
scan_data/                      # SCAN train/test splits
SPECTRAL_GEOMETRY_RESULTS.md    # Detailed spectral geometry results
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy, Matplotlib, SciPy

## Usage

Run a learning rate sweep with commutator tracking:
```bash
python scan/lr_sweep.py
python dyck/lr_sweep.py
```

Analyze commutator defect and generate figures:
```bash
python scan/commutator_analysis.py
python dyck/commutator_analysis.py
```

Run causal intervention experiments:
```bash
python scan/intervention.py
python dyck/intervention.py
```

Run spectral geometry analysis:
```bash
python scan/spectral_geometry.py
python dyck/spectral_geometry.py
```

## Citation

If you use this code, please cite:

```bibtex
@article{xu2026earlywarning,
  title={Early-Warning Signals of Grokking via Loss-Landscape Geometry},
  author={Xu, Yongzhong},
  year={2026},
  eprint={2602.16967},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2602.16967}
}
```

## License

MIT
