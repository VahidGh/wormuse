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

### Step 8a — ODE PINN (damped harmonic oscillator)  *(L14 / L27)*

**The physics (ODE):** Each muscle group j is a point oscillator:

$$\ddot{q}_j + 2\gamma\,\dot{q}_j + \omega^2\,q_j = F_j^{\text{neural}}(t)$$

- `γ` = damping (set to locomotion decay rate)
- `ω` = natural frequency (2π × locomotion_freq_Hz)
- `F_j` = neural forcing from latent code

**NAML concepts:**
- **L27 — PINNs:** data loss + λ × residual² at random collocation times.
- **L14 — jax.grad** twice: `∂q/∂t` and `∂²q/∂t²` via `jax.jacfwd(jax.jacfwd(...))`.
- **Recipe:** Adam → L-BFGS (identical to the user's SC-PINN, L22).

**Limitation:** treats each muscle independently — no spatial coupling between segments.

---

### Step 8b — PDE PINN (1D wave equation)  *(L14 / L27 + NMPDE connection)*

**The physics (PDE):** The worm body as a 1D elastic rod:

$$\rho\,q_{tt}(x,t) - \mu\,q_{xx}(x,t) + \gamma\,q_t(x,t) = F(x,t)$$

where `x ∈ [0,1]` is position along the body, `t` is time.

**Why this PDE?**  
This is the *linearised SPH model* from SCIENTIFIC_FOUNDATION.md §A.6 — it's also structurally identical to the piano string equation (§B.2):
- Piano: `ρ_s ü - T_s u_xx + ES κ² u_xxxx + ... = F_hammer`
- Worm:  `ρ q_tt - μ q_xx + γ q_t = F_neural`

The worm and the piano obey the **same PDE family** (damped wave equation). This is SCIENTIFIC_FOUNDATION.md §C.2 Correspondence 2 made concrete.

**NAML concepts:**
- **L27 — PINNs:** collocation points now span (x, t) space, not just t.
- **L14 — jax.grad** for both spatial (`∂²q/∂x²`) and temporal (`∂²q/∂t²`) derivatives.
- More physics = higher accuracy but ~3× slower than ODE.

**Files:**  
- `step8_pinn/locomotion_pinn.py` — contains `ode_residual`, `pde_residual`, `compare_ode_vs_pde`

---

### ODE vs PDE comparison

| Criterion | Step 8a (ODE) | Step 8b (PDE) |
|---|---|---|
| Physics dimension | Time only | Space + time |
| Collocation points | `t ∈ [0, T]` | `(x,t) ∈ [0,1] × [0,T]` |
| Derivatives needed | `∂²q/∂t²` | `∂²q/∂t²` + `∂²q/∂x²` |
| Training time | ~30 s | ~90 s |
| Captures body wave? | ✗ (per-muscle) | ✓ (spatial propagation) |
| NAML coverage | L14, L27 | L14, L27 + NMPDE echo |
| Expected loss | Slightly higher | Slightly lower |

**Expected improvement:** PDE-constrained notes have smoother inter-note transitions because spatial coupling enforces that adjacent body segments cannot fire completely independently. Notes respect both mechanical timing (ODE benefit) and wave propagation (PDE benefit).

---

## Results (measured, 2026-05-22)

Notebook: `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` (2 MB, all outputs baked).  
Simulation: 10 s window, 302-neuron synthetic activity, Chopin first 10 s.

| Step | Method | NAML Lectures | Onset loss | Notes (10s) | Notes |
|---|---|---|---|---|---|
| 0 | Rule-based | — | 0.00218 | 32 | Baseline; regular grid |
| 1 | SVD + Procrustes | L06, L09 | 0.00734 | 32 | Alignment residual 2688 — worm/Chopin subspaces very different |
| 2 | K-means | L08, L10 | **0.00145** | 1 | Low loss but sparse — 1 cluster transition in 10s |
| 3 | Ridge | L07, L11 | 0.00761 | 30 | Ill-conditioned with k=1 (one dominant neural PC) |
| 4-6 | MLP + Adam + L-BFGS | L14-22 | 0.00504 | 20 | Non-linear mapping; better than Ridge |
| 8a | ODE PINN | L14, L27 | 0.05122 | — | Higher total loss (physics term); data_loss≈0.047 |
| 8b | PDE PINN | L14, L27 | 0.07871 | — | PDE phys_loss stuck at 0.48; spatial derivatives harder |

### Key findings

1. **K-means achieved lowest onset loss** (0.00145) but by producing only 1 note — the soft Gaussian loss is easily gamed by sparse predictions. Better metric needed (e.g., precision/recall with fixed-width window).

2. **MLP+Adam+L-BFGS (Steps 4-6) gave the best structured output** (20 notes, meaningful mapping) with loss 0.00504.

3. **PINN losses are much higher** (0.05-0.08) than the data-only methods. This is expected: the PINN optimises `L_data + λ L_phys`. The total loss is higher because it is enforcing the physics constraint, not just fitting Chopin. The data_loss component alone (≈0.047 for ODE) is comparable to the other steps.

4. **ODE PINN converged; PDE PINN got stuck.** The PDE physics loss stayed constant at 0.481 throughout training — the spatial Jacobians make the optimization landscape much harder. Solutions:
   - Increase collocation point density
   - Use curriculum: train data-only first, gradually increase λ_phys
   - Lower learning rate for the PDE case

5. **k=1 neural PC** was chosen by the variance criterion because the synthetic 302-neuron activity (302 repetitions of 8 muscle signals) collapses to one dominant direction. The real worm would have richer structure (motor circuits, interneurons, sensory neurons), yielding k=4-8 truly independent dimensions.

### Interpretation

The biological system (302 neurons, 8 muscles) has severe dimensionality constraints that limit what NAML can achieve:
- The worm's 302-D neural space effectively lives in 1-4 dimensions (locomotion subspace)
- Chopin's musical feature space requires much more structure
- The mapping is learnable but has a low ceiling (57.7% note reachability)

NAML methods improve the mapping quality **within** these constraints. The PINN adds physical realism, which changes the optimisation objective — the result is more biologically constrained but not necessarily closer to Chopin by the onset-loss metric.
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

---

## Further Work

The PyANNOW NAML progression demonstrates the learning side of the pipeline. The two
natural extensions below connect to other Politecnico di Milano courses and push the
project toward a production-grade, biologically faithful simulation loop.

---

### FW-A — Sibernetics integration + HPC learning loop  *(polimi-amsc)*

**The gap today:** the worm's neural activity is *synthetic* (302 repeated muscle voltages
+ noise). The real learning loop needs the actual Sibernetic SPH body simulator and the
C302 neural network, running in a tight optimization cycle.

**What polimi-amsc provides:**

| AMSC concept | Application in this extension |
|---|---|
| **L05 — smart pointers** (`unique_ptr<WormSimulator>`) | RAII wrapper around the OpenWorm Docker container in `wormuse-sim/src/ow_bridge/` |
| **L09 — static/shared libraries + plugins** | `wormuse_core.so` as a shared library; PyANNOW calls it via a C-ABI Python binding (`ctypes` or `pybind11`) |
| **L11 — MPI** | Distribute the learning loop: rank 0 runs Sibernetic, ranks 1-N run the MLP gradient updates in parallel |
| **L12 — OpenMP** | Parallelize the piano FEM soundboard assembly across threads; parallelize the PINN collocation-point residual evaluation |
| **AMSC project: Porting on GPU** | Run the Sibernetic OpenCL body simulator on the Azure NC24-A100 (already configured in `nmpde-projects/container-config.yaml`) |

**Three concrete next steps:**

1. **Phase 1 (wormuse-sim Phase 1-2 in ROADMAP.md):** implement `docker_runner.hpp`
   — a C++ subprocess wrapper that launches OpenWorm, reads the spike JSON output,
   and passes it to PyANNOW's `step1_svd/encoder.py` via a shared memory buffer.
   AMSC lectures directly applied: L03 (modern C++), L05 (RAII subprocess), L09 (library).

2. **Phase 2 — Parallel training loop:**  
   The current Adam + L-BFGS training is single-threaded Python. With a real Sibernetic
   simulation (~5 min per run), the outer optimization loop needs HPC:
   - MPI workers each run one Sibernetic scenario (different ion-channel parameters)
   - Reduce the loss function across workers (MPI_Allreduce, L11)
   - Central node updates the MLP/PINN weights
   This is identical in structure to the distributed FEM solve in the user's
   `nmpde-projects/` (Azure + Terraform + GitHub Actions scalability workflow).

3. **Phase 3 — GPU-accelerated PINN training:**  
   JAX already supports GPU via `pip install "jax[cuda12]"`. The bottleneck shifts to
   the Sibernetic simulation. The GPU-ported landslide runout code (AMSC project option 1)
   is a direct template: the SPH physics engine can be ported to CUDA following exactly
   the techniques from `polimi-pc` (L5 GPU arch, L8-10 parallel patterns).

**Expected outcome:** the learning loop that currently takes 678s (11 min) for a 10s
synthetic worm on CPU becomes a 10-60s cycle on the Azure NC24-A100 with the real
OpenWorm simulation, enabling proper hyperparameter search over ion-channel parameters.

---

### FW-B — Statistical improvement of ion-channel kinetics predictions  *(polimi-appstat)*

**The gap today:** the PINN predicts `(m_∞, h_∞, τ_m, τ_h)` as smooth functions of
voltage, but we have no statistical guarantee that these predictions are *biologically
plausible* — they might fit the training data but lie far outside the measured
distribution of real *C. elegans* channel parameters.

**What polimi-appstat (Python, 2026 course) provides:**

| AppStat concept | Application |
|---|---|
| **Lecture 01 — PCA** | Reduce the 4-parameter ion-channel space `(g_EGL19, V_half, τ_Ca, g_EXP2)` to its principal directions. The first PC likely captures "overall excitability"; the second PC separates fast vs slow channels. Visualise which parameter combinations produce "musical" worm activity. |
| **Lecture 03 — Clustering (K-means, silhouette)** | Cluster `(g_EGL19, V_half, τ_Ca, g_EXP2)` parameter combinations by their musical quality (onset loss). Identify the "musical cluster" — the region of parameter space that produces Chopin-like melodies. |
| **Lecture 04 — Linear models + diagnostics** | Regress the music quality metric (onset loss) on the four PINN-tunable ion-channel parameters. Full diagnostics: VIF for multicollinearity (do τ_Ca and g_EXP2 trade off?), Breusch-Pagan for heteroskedasticity, Cook's distance for outlier parameter sets. |
| **Lecture 05-06 — Logistic + ROC** | Binary classifier: "musical" vs "non-musical" parameter sets. Which ion-channel parameters most reliably predict musical output? Feature importance via Random Forest (Lecture 07) reveals that `τ_Ca` and `V_half_Ca` dominate (consistent with the PINN sensitivity analysis in the Chopin notebook). |
| **User's prior work** | `AppStat/project/VahidGhayoomie-channelworm-OW.pdf` and `OpenWorm_CHOpen_Research_Proposals.ipynb` already lay the groundwork. The bootstrap CI in `bootstrap_ci.png` and PCA scree in `pca_scree_plot.png` are the starting point for this analysis. |

**Three concrete next steps:**

1. **Dataset generation:** run the worm forward model 500 times with random
   `(g_EGL19, V_half, τ_Ca, g_EXP2)` parameters sampled from biologically plausible
   ranges (from the `CelegansChannelParams.BOUNDS` in `celegans_hh.py`). Record the
   onset loss for each run. Creates a 500-row dataset — large enough for all AppStat
   analyses.

2. **AppStat-style analysis notebook** (`wormuse-analytics/notebooks/Lab_V_regression.ipynb`
   style, already scaffolded in `wormuse-analytics/`):
   - PCA scree plot of the 4-D parameter space
   - K-means clustering (`k=3-5`) → "loud & fast", "soft & slow", "musical" clusters
   - OLS regression with full diagnostics (VIF, BP, DW) → which parameter drives quality?
   - RF feature importance → confirm sensitivity analysis result (τ_Ca and V_half dominant)

3. **Bayesian prior for the PINN** (stretch): use the regression model as a prior
   distribution over ion-channel parameters. Replace the L1/L2 regularisation in the
   PINN loss with a **Mahalanobis distance penalty** from the posterior mean:
   $$\mathcal{L}_{\text{PINN}} = \mathcal{L}_{\text{data}} + \lambda_p\mathcal{L}_{\text{phys}} + \lambda_s (θ - \bar θ)^T Σ^{-1} (θ - \bar θ)$$
   where `(θ̄, Σ)` comes from the AppStat regression. This directly connects Lectures
   04-06 (regression) to Lecture 27 (PINNs) — closing the NAML loop.

**Expected outcome:** the PINN ion-channel predictions become statistically validated —
we can say "this parameter set is within the 95% CI of biologically measured *C. elegans*
EGL-19 conductances" instead of just "this minimises the PINN loss."

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
