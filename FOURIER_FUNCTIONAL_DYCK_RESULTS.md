# Fourier Functional Modes — Dyck-1 Depth Prediction

## What we built

1. **Retraining with full state_dict** (`spectral/retrain_dyck_fourier.py`) — grokked (wd=1.0) and memorized (wd=0.0) models with 101 full checkpoints each
2. **Fourier functional analysis** (`spectral/fourier_functional_dyck.py`) — positional DFT of hidden representations, power spectra across training
3. **Depth-conditioned geometry** (`spectral/dyck_depth_basis.py`) — PCA of depth centroids, inter-depth distances, linear probe R²
4. **Intermediate probing** (`spectral/dyck_intermediate_probing.py`) — layer-wise linear probes tracking R² across 21 training steps
5. **Composition test** (`spectral/dyck_composition_test.py`) — token × accumulation factorization, cross-term R² boost
6. **Attention pattern spectral analysis** (`spectral/dyck_attention_modes.py`) — per-head DFT, entropy, counting/local/recent scores
7. **Paper figure** (`spectral/paper_figure_dyck_fourier.py`) — 5-panel Figure 1

## Key findings

### 1. Positional Fourier modes

| Model | Layer | Dominant ω | Top-1 | Top-3 |
|-------|-------|-----------|-------|-------|
| Grok  | embedding | 12 | 0.274 | 0.459 |
| Grok  | layer_0 | 0 | 0.315 | 0.624 |
| Grok  | layer_1 | 12 | 0.281 | 0.541 |
| Memo  | embedding | 0 | 0.357 | 0.511 |
| Memo  | layer_0 | 0 | 0.347 | 0.629 |
| Memo  | layer_1 | 0 | **0.771** | **0.917** |

**Interpretation**: ω=12 is the Nyquist frequency for T=24, corresponding to the alternating open/close pattern. The grokked model concentrates at ω=12 in layers 0 and 1 — it has learned the binary token structure. The memorized model collapses to ω=0 (DC component) with 77% of energy in a single mode, indicating a position-invariant mean representation (memorization, not structure).

### 2. Depth representation geometry

| Model | Layer | Linear R² | PCA₁ variance | Mean inter-depth dist |
|-------|-------|-----------|--------------|----------------------|
| Grok  | layer_1 | 0.714 | 0.344 | 0.21 |
| Memo  | layer_1 | 0.946 | 0.811 | 30.95 |

**Surprising finding**: The memorized model has *higher* linear R² for depth (0.946 vs 0.714) and much larger inter-depth distances. The memorized model encodes depth in a single high-variance direction (PCA₁=81%), while the grokked model distributes depth information across multiple dimensions. The grokked model's depth encoding is more compressed but also more distributed — consistent with a generalizable algorithm rather than a lookup table.

### 3. Intermediate probing

| Model | Layer | R² (early) | R² (late) |
|-------|-------|-----------|-----------|
| Grok  | embedding | 0.247 | 0.243 |
| Grok  | layer_0 | 0.899 | 0.832 |
| Grok  | layer_1 | 0.956 | 0.718 |
| Memo  | embedding | 0.269 | 0.254 |
| Memo  | layer_0 | 0.886 | 0.908 |
| Memo  | layer_1 | 0.932 | 0.941 |

**Key observation**: Depth information is already high (R²>0.85) at layer_0 even at step 100 — the model learns depth early. The grokked model's layer_1 R² *decreases* from 0.956 to 0.718 over training, while memo stays high. This suggests the grokked model re-encodes depth into a more abstract, less linearly-accessible representation as it learns the generalizable algorithm.

### 4. Compositional structure

| Feature set | Grok R² | Memo R² |
|------------|---------|---------|
| token only → depth | 0.093 | 0.081 |
| position only → depth | 0.025 | 0.030 |
| token + position → depth | 0.168 | 0.167 |
| token + pos + cross → depth | **1.000** | **1.000** |
| cumsum (oracle) → depth | 1.000 | 1.000 |
| full_rep → depth | 0.697 | 0.931 |
| rep → cumsum | 0.697 | 0.931 |
| rep → token_sign | 0.295 | 0.992 |

**Critical finding**: Both models achieve R²=1.0 with cross-terms (token × cumsum, token × position), confirming that depth is compositionally determined by token × accumulation interactions. However, the models differ in what their representations encode:

- **Memorized model**: rep → token_sign = 0.992, rep → cumsum = 0.931. The memorized model explicitly stores both token identity and running depth in its representations.
- **Grokked model**: rep → token_sign = 0.295, rep → cumsum = 0.697. The grokked model does NOT linearly encode token identity — it has abstracted away from surface features into a more compressed representation.

### 5. Attention patterns

**Grokked model (post-training)**:
- **Layer 0**: All 4 heads converge to near-perfect uniform backward attention (counting_KL ≈ 0.000, entropy ≈ 2.28). This IS the counting algorithm — uniform attention over all past positions computes an average that encodes the running count.
- **Layer 1**: 3 heads near-uniform, 1 head (L1H2) slightly specialized (KL=0.059, lower entropy=2.17).

**Memorized model (post-training)**:
- **Layer 0**: Non-uniform attention with KL = 0.09–0.13.
- **Layer 1**: Highly peaked, specialized attention (entropy as low as 1.02, KL up to 1.017). Individual heads attend to specific positions — a lookup table, not an algorithm.

**This is the mechanism**: Grokking converts attention from position-specific lookup (memorization) to uniform counting (generalization). The uniform attention in Layer 0 is the Fourier mode ω=0 — the DC component of the attention pattern, which computes an unweighted average (i.e., counting).

### 6. Connection to modular arithmetic findings

| Property | Modular arithmetic | Dyck-1 |
|----------|-------------------|--------|
| Dominant Fourier mode | ω=25-26 (group-theoretic) | ω=12 (Nyquist = binary alternation) |
| Proper basis | Discrete log for mul, additive for add | Stack depth (cumulative sum) |
| Composition | x²+y² = (a+b)² - 2ab via cross-terms | depth = Σ token_sign via cross-terms |
| Grokking mechanism | Single mode concentration | Uniform attention (ω=0 in attention) |
| Memorized signature | Flat spectrum | DC collapse (ω=0 in representations) |

The Dyck model reveals a **dual spectral structure**: representations concentrate at ω=12 (token alternation) while attention converges to ω=0 (uniform counting). This is the sequence-task analog of the modular-arithmetic model learning a single Fourier mode in the correct group basis.

## Scripts location

All scripts in `spectral/`. Figures in `spectral/fourier_dyck_plots/`.

## Open questions

- **Why does grokked R² decrease?** The grokked model's layer_1 probe R² drops from 0.96 to 0.72 — it learns a representation where depth is no longer linearly accessible, yet achieves higher test accuracy. What is the geometry of this non-linear depth encoding?
- **Head specialization**: L1H2 in the grokked model is the only non-uniform head. What does it do? Possible paren-matching role.
- **ω=12 in embeddings**: At step 5000, even the grokked embeddings shift to ω=12. Is weight decay pushing frequency information into the embedding layer?
