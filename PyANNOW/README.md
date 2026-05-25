# PyANNOW

**The NAML sub-project.** Python / JAX / Flax composer that maps *C. elegans* 302-neuron
activity to Chopin piano music using a 10-step NAML progression — **and its inverse**:
extracting recurring musical patterns from Chopin and driving the worm's 96 body-wall
muscles to produce a live dance visualisation.  The **ion-channel PINN is the scientific
centrepiece** (see [`../ION_CHANNELS.md`](../ION_CHANNELS.md)); the best audio result
(v1.0.0) is the **Step 9b physics-residual MLP on the full 229 s piece**, synthesised
with `render_string_v2` (3 detuned strings + hammer transient + room reverb).

`PyANNOW` ≈ **Py**thon **A**rtificial **N**eural-channel **N**etwork **O**rchestrator for **W**ormuse.

*Current version: **v2.0.0***

### Quick demo (v2.0.0)

```bash
# From repo root — serves PyANNOW/ at localhost:7432
python3 -m http.server 7432 --directory PyANNOW
# Then open:
#   http://localhost:7432/presentation/index_v2.html   ← 24-slide NAML presentation
#   http://localhost:7432/worm_dance/worm_dance.html   ← standalone worm dance
```

---

## Layout

```
PyANNOW/
├── pyproject.toml
├── src/pyannow/
│   ├── ion_channels/           HH kinetics (EGL-19, EXP-2, NCA-1/2, SHK-1, UNC-2)
│   ├── composer/               run_forward_fast: 96-cell Boyle 4×24 worm simulation
│   ├── step1_svd/              RSVD encoder + Procrustes alignment
│   ├── step2_clustering/       PCA + K-means motor primitives (+ silhouette K*)
│   ├── step3_regression/       Ridge regression composer
│   ├── step4_ffnn/             JAX/Flax MLP composer
│   ├── step5_training/         Adam mini-batch trainer (NAML Lab 10)
│   ├── step6_lbfgs/            L-BFGS polish (NAML L22)
│   ├── step8_pinn/             ODE + PDE PINN locomotion constraints
│   ├── targets/midi_target.py  MIDI parser, onset metrics, calibrated_onset_detect
│   └── training/cv.py          Time-series CV + blocked bootstrap CI
├── chopin_score_net/           NB04 module: Fourier→residual MLP→piano roll
├── notebooks/
│   ├── 02_chopin_worm_optimizer.ipynb        Worm optimizer + audio playback
│   ├── 03_pyannow_naml_progression.ipynb     Full 10-step NAML progression (v1)
│   ├── 04_chopin_score_net.ipynb             Fourier time-net reference ceiling
│   ├── 05_pyannow_step9b_audio.ipynb         v1.0.0: Step 9b on full piece + v2 piano
│   ├── 06_chopin_patterns_worm_dance_v2.ipynb  ← v2.0.0: inverse pipeline
│   └── 06_pyannow_v2_patterns.ipynb          Pattern-bank builder (lightweight)
├── worm_dance/                 ← v1.2.0 / v2.0.0: live dance visualizer
│   ├── worm_dance.html         Self-contained JS visualizer (body wave + circuit)
│   ├── chopin.wav              Chopin nocturne audio (auto-loaded)
│   └── data/
│       ├── worm_dance_data.json          Pre-built pattern bank (K*=8 poses × 96 muscles)
│       └── worm_forward_v2_cache.npz     HH simulation cache (~370 MB)
├── docs/
│   ├── PyANNOW_NAML_progression.md      Living doc: v1 F1 scores + v2 pipeline
│   └── boyle_architecture_map.html      Biology→NN table + piano keyboard diagram
├── presentation/
│   ├── index.html              v1.0.0 Reveal.js slides (10 NAML steps)
│   ├── index_v2.html           v2.0.0 split-screen presentation (24 slides + live dance)
│   ├── QA_v2.md                23 NAML Q&A pairs for the v2 presentation
│   └── SCENARIO_v2.md          20-minute speaking script
└── tests/
    ├── score_f1_quick.py        78-second F1 benchmark (no PINN/JAX)
    └── test_*.py                pytest unit tests
```

## Stack

Per the NAML labs (Lab04-10 are JAX-based; Lab10 introduces Flax + Optax):

```bash
pip install jax jaxlib flax optax \
            numpy scipy matplotlib seaborn \
            pandas scikit-learn \
            jupyter
# Optional GPU:
# pip install "jax[cuda12]"
```

Python 3.10-3.13.

## v1.x — Forward pipeline: Worm → Music (notebook 03, v0.9.0 measured F1)

| Step | Method | NAML / AppStat ref | F1 | Notes |
|---|---|---|---|---|
| 0 | Deterministic body-wave | — | 0.2170 | Baseline |
| 1a | SVD / RSVD | L06 Eckart-Young | 0.1861 | Unsupervised — no Chopin labels |
| 1b | SVD + Procrustes | L06/L09 | 0.0952 | Unsupervised |
| 2 | PCA + K-means | L08/L10, AppStat LabIII | 0.1086 | Unsupervised |
| 3 | Ridge + Lab V diagnostics | L07/L11, AppStat LabV | 0.2860 | First supervised win |
| 4-6 | MLP + Adam + L-BFGS | L14-22 | ~0.25 | JAX; 50-ep quick-test est. |
| 7 | RandomForest | AppStat Lec07 | 0.8721 | OOB=0.829, 13 s |
| 8 | ODE/PDE PINN | L14/L27 | SKIPPED | Set SKIP_PINN=False (~5 min) |
| **9** | **Worm+Time hybrid MLP** | **AppStat Lec06, NB04** | **0.8786** | **3.2 s, best step** |

## v2.0.0 — Inverse pipeline: Music → Worm Dance (notebook 06)

| Stage | Method | NAML ref | Output |
|---|---|---|---|
| 1 | Piano-roll M ∈ ℝ^{56×11711} | — | Binary pitch×time matrix |
| 2 | RSVD (Eckart-Young) k=12 | L06–L09 · Lab01 | U_k, σ_k, V_k (90% var.) |
| 3 | K-means on V_k, K*=8 | L10 · Lab02 | Musical state labels |
| 4 | Silhouette score | AppStat · Lab02 | K*=8 optimal clusters |
| 5 | Pearson excitability | AppStat | Pattern ranking (r≤0.88) |
| 6 | lstsq / pseudoinverse | L07/L09 · Lab03 | W_nm: neural→96-muscle map |
| 7 | Synthetic body wave | — | 96-muscle pose per pattern |
| 8 | Live JS visualizer | — | `worm_dance.html` dance demo |

## Lecture map (NAML + AppStat)

### Linear algebra & low-rank (NAML L06–L09)

| Lecture | Concept | Where |
|---|---|---|
| L06 Eckart-Young | Best rank-k = truncated SVD | `step1_svd/encoder.py` |
| L08 PCA | 302-D → k PC directions | `step2_clustering/motor_primitives.py` |
| L09 Pseudoinverse | Least-squares decoder from latent | `step1_svd/procrustes.py` |

### Regression & regularization (NAML L07/L11, AppStat LabV)

| Lecture | Concept | Where |
|---|---|---|
| L07 Normal equations | Ridge: `(Z^T Z + λI)^{-1} Z^T C` | `step3_regression/ridge_composer.py` |
| L11 Ridge | σ/(σ²+λ) shrinkage view | `step3_regression/ridge_composer.py` |
| AppStat Lab V | VIF, Durbin-Watson, Breusch-Pagan, LassoCV | notebook 03 Step 3 cell |

### Optimization (NAML L18–L22)

| Lecture | Concept | Where |
|---|---|---|
| L19-20 Adam | Per-parameter adaptive LR | `step5_training/adam_trainer.py` |
| L22 L-BFGS | Hessian approximation polish | `step6_lbfgs/lbfgs_polish.py` |

### Autodiff & NNs (NAML L14–L17)

| Lecture | Concept | Where |
|---|---|---|
| L14 Autodiff | `jax.grad` reverse-mode | `step4_ffnn/jax_composer.py` |
| L15 Activations | tanh / ReLU | `step4_ffnn/jax_composer.py` |
| L16-17 FFNN | Flax `nn.Dense`, Xavier init | `step4_ffnn/jax_composer.py` |

### Classification (AppStat Lec06/07)

| Lecture | Concept | Where |
|---|---|---|
| AppStat Lec07 | RandomForest, OOB, permutation importance | notebook 03 Step 7 |
| AppStat Lec06 | MLP classifier, onset probability | notebook 03 Step 9 |
| NB04 | Fourier time embeddings → score reconstruction | `chopin_score_net/`, notebook 04 |

### PINNs (NAML L27)

| Lecture | Concept | Where |
|---|---|---|
| L27 PINNs | Data loss + physics residual (ODE/PDE) | `step8_pinn/locomotion_pinn.py` |
| L14+L27 | `jax.grad` twice for ∂²q/∂t², ∂²q/∂x² | `step8_pinn/locomotion_pinn.py` |

## Phases

- **Phase 3** — `ion_channels/` PINN trained on synthetic HH. Adam→L-BFGS pipeline working end-to-end.
- **Phase 6** — full composer chain: PINN → encoded states → MIDI. Three audibly distinct scenarios from three ion-channel parameter sets.
- **Phase 8** — (stretch) inverse problem: given a target spectrogram, solve for ion-channel parameters.

See [../ROADMAP.md](../ROADMAP.md) for the full plan.

## Conventions

- **JAX-functional style** — pure functions, immutable arrays, explicit RNG via `jax.random.PRNGKey`.
- `@jax.jit` every hot inner loop. Profile with `jax.profiler`.
- Use **Flax `nn.Module`** for any model with parameters; **plain functions** for stateless math.
- Optimizer state via `optax` triples (`init`, `update`, `apply_updates`).
- All matplotlib styling matches the NAML labs (matter-of-fact, no fancy themes).

## When stuck

- `polimi-naml` skill: `svd-and-rsvd.md`, `pca-and-projections.md`, `optimization-methods.md`, `neural-networks.md`, `pinns.md`.
- The 5 NAML code snippets (svd compression, RSVD, PCA from scratch, optimization zoo, PINN-1D-Poisson) are working references.
- `polimi-appstat` skill for statistical diagnostics of training curves and outputs.
- The user's `naml-ion-channel-pinn/` (in `Numerical_analysis_forML/`) is the **conceptual** reference — read its README and notebooks for the architecture rationale. We do **not** depend on it as a library (per design).
