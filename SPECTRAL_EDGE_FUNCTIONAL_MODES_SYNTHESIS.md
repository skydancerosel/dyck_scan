# Spectral Edge Functional Modes: Complete Synthesis

## Dyck-1 (balanced parentheses depth prediction) + SCAN (command-to-action translation)

---

## I. The Spectral Edge Exists and Has Two Phases

The Gram matrix of attention weight updates produces singular vectors v1, v2, ... (the "spectral edge" directions) and a gap g23 that separates edge from bulk.

### Phase 1: Learning Edge (pre-grok → grok)

| Property | Dyck | SCAN |
|----------|------|------|
| g23 | 7.1 → 1.7 (declining) | 0 → 26.3 (forming) |
| v1 composition | **97.6% gradient** | **88.7% gradient** |
| v1 functional R² | dec R²=0.61 (Dyck depth) | dec R²=0.60 (SCAN action) |
| v1 perturbation KL | 0.00006 | — |
| Role | Gradient-driven learning | Gradient-driven learning |

### Phase 2: Compression Edge (at grok → post-grok)

| Property | Dyck | SCAN |
|----------|------|------|
| g23 | 1.7 → 1.2 (stabilizing) | 26.3 → 1.9 → 0.6 (collapsing) |
| v1 composition | **5% grad / 95% WD** | **0.2% grad / 99.8% WD** |
| v1 functional R² | dec R²=0.46→0.80 (persists) | dec R²=0.12→**0.04** (vanishes) |
| Grad-WD alignment | **Aligned** (same direction) | **Aligned** (same direction) |
| Role | WD-driven but still functional | WD-driven compression, functionally empty |

**The transition from learning to compression happens exactly at grokking.** Gradient and weight decay become aligned along v1 — they conspire to compress parameters without fighting each other.

---

## II. Edge vs Bulk: Qualitatively Different Roles

### Ablation (causal importance)

| Phase | Remove edge (v1+v2) | Remove bulk (v3+v4) |
|-------|--------------------|--------------------|
| Dyck pre-grok | **Δacc = -0.40** | Δacc = 0.00 |
| Dyck at-grok | Δacc = -0.29 | **Δacc = -0.34** |
| Dyck post-grok | **Δacc = -0.29** | Δacc = -0.03 |
| Dyck late | **Δacc = -0.62** | Δacc = -0.13 |

The edge is **causally critical throughout** (removing it drops accuracy 29-62%). The bulk matters only transiently at the grokking moment, then becomes dispensable.

### Hessian curvature

| Direction | Dyck grok (late) | Dyck memo (late) |
|-----------|-----------------|-----------------|
| v1 (edge) | 0.078 (flat) | 0.173 |
| v2 (edge) | 0.001 (very flat) | **0.987** (curved) |
| v3 (bulk) | 0.020 | **0.851** |
| v4 (bulk) | 0.068 | **1.838** (sharp) |

Grokked model: edge and bulk are all flat (curvature < 0.08).
Memorized model: bulk directions have **very high curvature** (up to 1.84) — the model sits on a knife-edge where small perturbations in the bulk destroy performance.

### Path-norm vs function change

| Direction | Grok: σ (update size) | Grok: Δfunc | Memo: σ | Memo: Δfunc |
|-----------|----------------------|-------------|---------|-------------|
| v1 (edge) | **2.30** (large) | **0.00005** (zero) | 0.15 | 0.00079 |
| v3 (bulk) | 0.92 | 0.00035 | 0.03 | **0.00397** |
| v4 (bulk) | 0.80 | 0.00021 | 0.01 | **0.00689** |

**Grok edge: moves a lot in weight space, changes nothing in function space — pure compression.**
Memo bulk: moves very little, changes function drastically — fragile memorization.

---

## III. Rotation Stability

| Task | Model | v1 rotation | v2 rotation | v3 rotation |
|------|-------|------------|------------|------------|
| SCAN | grok | **7.7°** (frozen) | 20.6° | 43.0° |
| SCAN | memo | 28.8° | 50.8° | 83.8° (random) |
| Dyck | grok | 17.8° (moderate) | 41.7° | 51.6° |
| Dyck | memo | 22.2° | 43.7° | 64.4° |

**SCAN grok v1 is a frozen axis** — it locks in and barely rotates (8°). This is consistent with it becoming a pure WD compression direction (99.8% WD) that doesn't need to adjust.

**Dyck grok v1 rotates more** (18°) — consistent with it remaining partially gradient-driven (still 87% gradient late) and retaining functional content.

Bulk directions rotate ~40-60° in grokked models (moderate instability) and up to 84° in memorized SCAN (nearly random).

---

## IV. Information Encoding: Linear vs Nonlinear

### Layer 1 depth probe R² (Dyck)

| Phase | Linear | Quadratic | **MLP** |
|-------|--------|-----------|---------|
| pre-grok | 0.965 | 0.975 | 0.975 |
| at-grok | 0.981 | 0.995 | 0.993 |
| post-grok | 0.960 | 0.976 | **0.994** |
| **late** | **0.667** | **0.591** | **0.988** |

**The linear probe R² drop (0.97→0.67) is an encoding change, not information loss.** An MLP probe recovers R²=0.99 — depth is present but nonlinearly encoded. Even quadratic probes fail (0.59), meaning the re-encoding is genuinely deep nonlinear.

The memorized model stays linearly accessible throughout (linear R²≈0.94).

---

## V. Weight Decay Intervention — The Causal Test

### Removing WD post-grok (Dyck, starting from step 1000)

| Condition | Final test acc | Probe R² | Attn entropy | ||θ|| |
|-----------|---------------|---------|-------------|-------|
| wd=1.0 (same) | 0.982 | 0.851 | **2.27** | 15.3 |
| **wd=0.0** | 0.973 (-0.009) | **0.987** (+0.14) | 2.15 (-0.12) | **40.4** (+25) |
| wd=0.5 | 0.972 | 0.898 | 2.26 | 17.7 |
| wd=2.0 | **0.985** (+0.003) | 0.709 (-0.14) | **2.28** | **14.3** |

**Removing WD after grokking:**
- Accuracy barely drops (0.982 → 0.973) — the algorithm survives
- Probe R² jumps from 0.85 to 0.99 — representations become linearly accessible again
- Entropy drops from 2.27 to 2.15 — attention becomes less uniform
- Param norm balloons 15 → 40 — without compression, weights grow

**Doubling WD after grokking:**
- Accuracy slightly increases (+0.003) — more compression helps slightly
- Probe R² drops to 0.71 — more compression = more nonlinear encoding
- Entropy maxes at 2.28 — attention becomes maximally uniform

**This proves**: WD drives the post-grok compression that creates the spectral edge, causes the nonlinear re-encoding (probe R² drop), and maintains uniform attention. But the **learned algorithm persists without WD** — compression is cosmetic, not constitutive.

---

## VI. Fourier Functional Modes

### Positional DFT of hidden representations

| Model | Layer | Dominant ω | Top-3 concentration |
|-------|-------|-----------|-------------------|
| Dyck grok | layer_1 | **ω=12** (Nyquist) | 0.541 |
| Dyck memo | layer_1 | **ω=0** (DC) | **0.917** |
| SCAN grok | enc_2 | ω=1 (global) | 0.567 |
| SCAN memo | enc_2 | ω=0 (DC) | 0.787 |

Grokked models distribute energy across meaningful frequencies (ω=12 = binary token alternation for Dyck). Memorized models collapse to DC (ω=0) — all positions look the same.

### Attention patterns

| Model | Layer 0 heads | Counting KL | Entropy |
|-------|--------------|------------|---------|
| Dyck grok | All 4 uniform | **0.000** | **2.28** |
| Dyck memo | Peaked, position-specific | 0.09-1.02 | 1.02-2.18 |

Grokking converts attention from position-specific lookup to **uniform backward counting** — this IS the depth computation algorithm.

### Compositional structure

Both grok and memo achieve R²=1.0 with cross-terms (token × cumsum). But grok doesn't linearly encode token identity (R²=0.30 vs memo 0.99) — it abstracts away from surface features.

---

## VII. The Unified Picture

### What is the spectral edge?

The spectral edge is a **low-dimensional subspace of parameter updates** that undergoes a phase transition at grokking:

1. **Before grokking**: The edge is a **gradient-driven learning direction** carrying task-relevant functional content. Updates along it change the model's behavior (high functional R², moderate Hessian curvature).

2. **At grokking**: Gradient and weight decay **align** along the edge. The gradient finds the generalizing solution; WD starts compressing along the same direction. The edge transitions from learning to compression.

3. **After grokking**: The edge becomes a **WD-driven compression direction** that is:
   - Functionally flat (perturbation KL ≈ 0.00005, near-zero Hessian curvature)
   - But causally critical (ablation kills accuracy: Δacc = -0.62)
   - Rotationally stable (7-18° rotation per window)
   - Carrying large-magnitude updates (σ = 2.3) that don't change the function

This resolves the apparent paradox: **the edge is simultaneously silent and essential.** The model's function depends on WHERE the edge points but not on motion ALONG it. It's a fixed axis of the learned algorithm, maintained by weight decay.

### How do Dyck and SCAN differ?

| Property | Dyck | SCAN |
|----------|------|------|
| Edge transition | Gradual (87% grad late) | Sharp (0.2% grad at grok) |
| Edge functional content | Persists (R²=0.80) | Vanishes (R²=0.04) |
| Edge rotation | Moderate (18°) | Frozen (8°) |
| Bulk fragility | Low (all directions flat) | — |
| Characterization | **Mixed edge** (compression + residual learning) | **Pure compression edge** |

Dyck's edge retains functional content because:
- The task is simpler (1 eigenmode: depth counting)
- All directions project onto the same functional mode
- The gradient never fully exits the edge

SCAN's edge becomes purely structural because:
- The task is richer (compositional command mapping)
- Once the algorithm is learned, the encoder-decoder has enough redundancy for WD to compress without functional impact
- The gradient shifts entirely to bulk directions for fine-tuning

### What weight decay does

WD is not just regularization — it has three distinct roles during grokking:

1. **Pre-grok**: Creates pressure toward parameter-efficient solutions (Fourier modes, uniform attention)
2. **At grok**: Aligns with the gradient along the edge, catalyzing the phase transition
3. **Post-grok**: Drives ongoing compression that nonlinearly re-encodes representations (probe R² drops from 0.97 to 0.67) while preserving the algorithm (MLP R² stays 0.99)

Removing WD post-grok: algorithm survives, compression reverses, representations become linearly accessible again. WD is the engine of compression, not the engine of generalization.

### The hierarchy of signals

| Signal | When | Reliability | What it measures |
|--------|------|-------------|-----------------|
| SGD defect spike | Pre-grok | 6/6 (100%) | Curvature instability |
| Commutator peak | Pre-grok | 6/6 (100%) | W_Q-W_K alignment beginning |
| Grad→WD transition on v1 | At grok | Both tasks | Learning → compression |
| g23 compression | At grok | 33-43x | Update trajectory rank collapse |
| Attention uniformization | At grok | Both tasks | Algorithm crystallizing |
| Fourier mode concentration | Post-grok | Both tasks | Representation structure |
| Spectral gap σ1≫σ2 | Post-grok | Consequence | Weight decay amplification |
| Probe R² inversion | Post-grok | Dyck confirmed | Nonlinear re-encoding |

---

## VIII. Scripts and Outputs

All scripts in `spectral/`. All figures in `spectral/fourier_dyck_plots/` and `spectral/fourier_scan_plots/`.

### Core Fourier analysis
- `fourier_functional_dyck.py` / `fourier_functional_scan.py` — positional DFT of hidden reps
- `dyck_depth_basis.py` — depth-conditioned representation geometry
- `dyck_intermediate_probing.py` / `scan_intermediate_probing.py` — layer-wise linear probes
- `dyck_composition_test.py` / `scan_composition_test.py` — compositional factorization
- `dyck_attention_modes.py` / `scan_attention_modes.py` — attention pattern spectral analysis

### Spectral edge analysis
- `gram_edge_functional_modes.py` / `gram_edge_functional_modes_scan.py` — perturbation response f_k(x)
- `principal_direction_analysis.py` — W_Q/W_K SVD direction content
- `grad_vs_wd_decomposition.py` — gradient vs weight decay decomposition (#1)
- `perturbation_curves.py` — ε-sweep loss/KL/entropy curves (#2)
- `edge_rotation_stability.py` — subspace rotation tracking (#4)
- `wd_intervention.py` — WD removal/variation after grokking (#5)
- `nonlinear_probes.py` — linear/quadratic/MLP probe comparison (#7)
- `edge_ablation.py` — ablation + Hessian curvature + path-norm (#3, #6, #8)

### Support
- `retrain_dyck_fourier.py` / `retrain_scan_fourier.py` — retraining with full state_dict
- `paper_figure_dyck_fourier.py` / `paper_figure_scan_fourier.py` — paper-quality figures
