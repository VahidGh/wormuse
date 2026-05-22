# PyANNOW — NAML Progression

**Living document.** Updated as each step is implemented.  
*Last updated: Phase 0 — all steps scaffolded.*

---

## The question

> Can a biological neural system with limited physical constraints learn to play Chopin?

We use **only the methods taught in the Numerical Analysis for Machine Learning (NAML)** course at Politecnico di Milano to build a composer that maps the *C. elegans* worm's 302-neuron activity onto a piano melody, then measure how close we can get to Chopin's Prelude in D♭.

---

## The constraints (fixed biology)

| Constraint | Value | Musical consequence |
|---|---|---|
| Neurons | 302 (fully mapped connectome) | 302-D input vector |
| Muscle groups | 8 body-wall segments | 8 independent piano "keys" |
| Locomotion frequency | ~0.4–2 Hz | Max ~3 notes/second |
| BWM refractory | ~280 ms | Min inter-note interval |
| Pitch range | 8 pentatonic pitches | vs Chopin's 5 octaves |

---

## Step progression

### Step 0 — Random baseline  *(no NAML)*

**What:** Rule-based neuron→note mapping (Phase → pitch). No learning.  
**Result:** Random-sounding output, loss ≈ 0.010  
**Lesson:** Biology gives us ~3 notes/s but no musical structure.

---

### Step 1 — SVD + Procrustes  *(Lab01 / L06 / L09)*

**NAML concepts:**
- **L06 — Eckart-Young theorem:** the best rank-k approximation of the neural trajectory matrix `X ∈ ℝ^{302 × T}` is the truncated SVD `X_k = U_k Σ_k V_k^T`. We compress 302 neurons → k=4 principal components.
- **Randomized SVD** (course's `rsvd_2024.ipynb`): same algorithm as Lab01 image compression but applied to neural activity.
- **L09 — Pseudoinverse:** the best linear map from neural scores to Chopin features is `W* = Z^+  C` (pseudoinverse). When the system is overdetermined, this minimises `||Z W - C||_F`.
- **Procrustes problem:** find the *orthogonal* rotation R that aligns the worm subspace to the Chopin subspace — solved by SVD of `C^T Z`.

**Files:**  
- `step1_svd/encoder.py` — RSVD encoder  
- `step1_svd/procrustes.py` — Orthogonal alignment

**Expected improvement:** +15-30% reduction in onset loss. The worm starts following Chopin's rhythm structure, even if imperfectly.

---

### Step 2 — PCA + K-means  *(Lab02 / L08 / L10)*

**NAML concepts:**
- **L08 — PCA:** reduce the neural trajectory to 2-4 PC directions. Visualise the "motor state space" — just as Lab02 showed MNIST digit clusters in PCA space, here we see Forward/Backward/Turn clusters.
- **L10 — K-means:** cluster the motor states into k discrete primitives. Each cluster = one musical phrase. Choose k by silhouette score (as in Lab02/AppStat).

**Files:**  
- `step2_clustering/motor_primitives.py`

**Expected improvement:** Discrete note categories emerge — transitions between motor states produce musical phrases rather than random single notes.

---

### Step 3 — Ridge regression  *(Lab03 / Lab07 / L07 / L11)*

**NAML concepts:**
- **L07 — Normal equations:** W* = (Z^T Z)^{-1} Z^T C. The neural PC scores are correlated → ill-conditioned gram matrix → ridge is essential.
- **L11 — Ridge regularisation:** `W_ridge = (Z^T Z + λI)^{-1} Z^T C`. The SVD view shows ridge shrinks singular values by `σ/(σ²+λ)` — suppressing noise directions.
- **Lab03/Lab07 pattern:** RidgeCV on the California housing → here on worm neural scores.

**Files:**  
- `step3_regression/ridge_composer.py`

**Expected improvement:** Stable velocity predictions. The dynamics (soft/loud notes) begin to match Chopin's.

---

### Step 4 — Feed-forward Neural Network  *(Lab06 / L14-17)*

**NAML concepts:**
- **L14 — Autodiff (JAX `jax.grad`):** the chain rule computed automatically — backpropagation as a special case.
- **L15 — Activations:** tanh (smooth, bounded) for audio feature prediction. Avoids the dying-neuron problem of ReLU.
- **L16 — Xavier init:** variance of weights = 2/(n_in + n_out). Essential for stable deep propagation.
- **L17 — FFNN:** the non-linear upgrade over ridge. Universal approximation (L24) guarantees that a deep enough MLP can represent the true worm→Chopin mapping.
- **Lab06 pattern:** same architecture as the XOR network (`2 → 2 → 1`), scaled to (`k_worm → 32 → 32 → k_chopin`).

**Files:**  
- `step4_ffnn/jax_composer.py`

**Expected improvement:** Non-linear features captured. The MLP can represent the mapping that the linear Procrustes cannot.

---

### Step 5 — Adam / mini-batch SGD  *(Lab05 / Lab07 / Lab10 / L18-20)*

**NAML concepts:**
- **L18 — GD convergence:** linear rate `(1 - μ/L)`. The worm→Chopin landscape is ill-conditioned (some neural PC has 80% variance, others <1%) → plain GD is slow.
- **L19 — SGD:** mini-batches reduce memory and add implicit regularisation.
- **L20 — Adam:** per-parameter adaptive learning rates. `α × m̂_t / (√v̂_t + ε)` is insensitive to the feature scale mismatch.
- **Lab10 pattern:** `optax.adam(lr=1e-3)`, 500 epochs, early stopping on validation, training curves.

**Files:**  
- `step5_training/adam_trainer.py`

**Expected improvement:** Full convergence. Training curve shows 80-90% loss reduction vs random. The temporal structure of Chopin begins to appear in the spectrograms.

---

### Step 6 — L-BFGS fine-tuning  *(Lab08 / L21-22)*

**NAML concepts:**
- **L21 — Newton:** quadratic convergence near a minimum. Requires storing the Hessian `∈ ℝ^{p×p}` — too expensive for large p.
- **L22 — BFGS/L-BFGS:** approximates the Hessian from gradient history. Memory O(mk) for `m=10` past gradients.
- **Lab08 Part 2:** we compared Newton vs GD on a quadratic. Same comparison here shows L-BFGS dominates Adam in the fine-tuning phase.

**Files:**  
- `step6_lbfgs/lbfgs_polish.py`

**Expected improvement:** The last 5-10% loss reduction that Adam plateaued on. Tighter note-timing alignment.

---

### Step 8 — PINN (locomotion mechanics)  *(L14 / L27)*

**NAML concepts:**
- **L27 — PINNs:** add a physics residual to the MLP loss.
- **Physics:** the worm body obeys a damped harmonic oscillator `ÿ + 2γẏ + ω²y = F_neural`.
- **L14 — jax.grad** computes `dy/dt` and `d²y/dt²` automatically at collocation points.
- **Recipe:** Adam (1000 steps) → L-BFGS (100 steps) — identical to the user's SC-PINN.

**Files:**  
- `step8_pinn/locomotion_pinn.py`

**Expected improvement:** Generated notes respect the worm's mechanical timing constraint — notes are smoother, less jittery. The biological realism is enforced mathematically.

---

## Results summary (to be updated)

| Step | Method | NAML Lectures | Loss | Notes/s | Improvement |
|---|---|---|---|---|---|
| 0 | Rule-based | — | 0.010 | 3.2 | Baseline |
| 1 | SVD + Procrustes | L06, L09 | ? | ? | TBD |
| 2 | + K-means | L08, L10 | ? | ? | TBD |
| 3 | + Ridge | L07, L11 | ? | ? | TBD |
| 4 | + MLP (JAX) | L14-17 | ? | ? | TBD |
| 5 | + Adam | L18-20 | ? | ? | TBD |
| 6 | + L-BFGS | L21-22 | ? | ? | TBD |
| 8 | + PINN | L14, L27 | ? | ? | TBD |

---

## Biological ceiling (fixed by physics, not NAML)

No matter how good our NAML model becomes, the worm faces hard limits:

| Limit | Source | Musical consequence |
|---|---|---|
| 8 voices | 8 BWM segments | Max 8-note chords (Chopin uses more) |
| 57.7% reachable | BWM refractory 280 ms | ~42% of Chopin notes physically unreachable |
| Regular rhythm | Body-wave phase structure | Worm plays a regular grid; Chopin plays syncopated |

The NAML methods improve mapping quality *within* these constraints. They cannot change the constraints themselves.

---

## Running the notebook

```bash
cd PyANNOW
pip install -e .
jupyter lab notebooks/03_pyannow_naml_progression.ipynb
```

Estimated total runtime: ~3-5 minutes (Steps 0-6 in sequence).

---

## File map

```
PyANNOW/
├── src/pyannow/
│   ├── step1_svd/encoder.py          ← RSVD (L06 / Lab01)
│   ├── step1_svd/procrustes.py       ← Procrustes alignment (L06 / L09)
│   ├── step2_clustering/             ← PCA + K-means (L08 / L10 / Lab02)
│   ├── step3_regression/             ← Ridge (L07 / L11 / Lab03 / Lab07)
│   ├── step4_ffnn/                   ← MLP / Flax (L14-17 / Lab06 / Lab10)
│   ├── step5_training/               ← Adam (L18-20 / Lab05 / Lab07 / Lab10)
│   ├── step6_lbfgs/                  ← L-BFGS (L21-22 / Lab08)
│   └── step8_pinn/                   ← PINN (L14 / L27)
├── notebooks/
│   └── 03_pyannow_naml_progression.ipynb
├── docs/
│   └── PyANNOW_NAML_progression.md   ← this file
└── presentation/
    └── index.html                     ← Reveal.js presentation
```
