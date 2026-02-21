# Early-Warning Signals of Grokking: SCAN and Dyck Experiments

**Paper**: [Early-Warning Signals of Grokking via Loss-Landscape Geometry](https://arxiv.org/abs/2602.16967)

Code and figures for studying the **commutator defect** as an early-warning signal for grokking on two sequence-learning tasks:

- **SCAN** compositional generalization (encoder-decoder transformer, ~1.5M params)
- **Dyck-1** depth prediction (causal decoder transformer, ~150k params)

## Key Results

The commutator defect --- a measure of loss-landscape curvature from the non-commutativity of successive gradient updates --- reliably rises before the onset of generalization. The lead time follows a **superlinear power law**:

| Task | Exponent (alpha) | R^2   | Data points |
|------|:-----------------:|:-----:|:-----------:|
| SCAN | 1.18              | 0.990 | 11          |
| Dyck | 1.13              | 0.908 | 14          |

Combined with prior results on modular arithmetic, all three task families exhibit alpha > 1, yielding advance warning windows of 90-97% at slow learning rates.

## Repository Structure

```
# SCAN experiments
scan_grok_sweep.py              # Grokking sweep across learning rates
scan_lr_sweep.py                # Learning rate sweep with commutator tracking
scan_lr_sweep_extended.py       # Extended sweep configuration
scan_commutator_analysis.py     # Commutator defect analysis and plotting
scan_generalization_dynamics.py # Generalization dynamics tracking
scan_intervention.py            # Causal intervention experiments
scan_pca_analysis.py            # Weight-space PCA analysis
scan_pc1_lr_experiment.py       # PC1 trajectory experiments
scan_lm_pilot.py                # Language model pilot experiments
scan_data/                      # SCAN train/test splits

# Dyck experiments
dyck_grok_sweep.py              # Grokking sweep across learning rates
dyck_lr_sweep.py                # Learning rate sweep with commutator tracking
dyck_lr_sweep_extended.py       # Extended sweep configuration
dyck_commutator_analysis.py     # Commutator defect analysis and plotting
dyck_generalization_dynamics.py # Generalization dynamics tracking
dyck_intervention.py            # Causal intervention experiments
dyck_pca_analysis.py            # Weight-space PCA analysis
dyck_pc1_lr_experiment.py       # PC1 trajectory experiments
dyck_lm_pilot.py                # Language model pilot experiments
dyck_lr1e2_diagnostic.py        # Diagnostic for lr=1e-2

# Cross-task
grok_pc1_lr_experiment.py       # Joint PC1 analysis

# Figures
scan_pca_plots/                 # SCAN figures
dyck_pca_plots/                 # Dyck figures
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy, Matplotlib, SciPy

## Usage

Run a learning rate sweep with commutator tracking:
```bash
python scan_lr_sweep.py
python dyck_lr_sweep.py
```

Run the extended sweep (multiple seeds, adjusted schedules):
```bash
python scan_lr_sweep_extended.py
python dyck_lr_sweep_extended.py
```

Analyze commutator defect and generate figures:
```bash
python scan_commutator_analysis.py
python dyck_commutator_analysis.py
```

Run causal intervention experiments:
```bash
python scan_intervention.py
python dyck_intervention.py
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
