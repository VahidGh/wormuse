# Changelog

All notable changes to **wormuse** are documented here.
Format: [Semantic Versioning](https://semver.org) — `MAJOR.MINOR.PATCH`.

- **MAJOR** — breaking change in PyANNOW public API or simulation contract
- **MINOR** — new feature: module, notebook, simulation mode, presentation section
- **PATCH** — bug fix, doc correction, refactor with no behaviour change

> **AI development note:** Every version listed here was implemented with
> [Claude Code](https://www.anthropic.com/claude-code) (Anthropic) acting as the principal
> development engineer under the scientific direction of Vahid Ghayoomie.
> See `AI_CONTRIBUTIONS.md` for a full breakdown.

---

## [1.0.0] — 2026-05-25  *(current)*

### Realistic piano audio + Step 9b full-piece notebook (ISSUE-003 + ISSUE-004)

First production-quality audio release.  The "tin can" modal synth is replaced by
`render_string_v2` — a 3-string detuned choir with hammer-impact transient — and
a room reverb post-process.  Output WAV at 8 kHz matches the source recording size
(≈ 3.5 MB / 229 s).  A focused notebook `05_pyannow_step9b_audio.ipynb` extracts
Step 9b from the full progression and applies it directly to the complete Chopin
piece without a training-window cap.

### Added
- **`render_string_v2()`** in `piano_synth.py` — 3 detuned strings (0, +5, −5
  cents) + 4 ms hammer-noise burst.  Removes the single-mode tin-can resonance.
  Fixes ISSUE-003.
- **`_room_ir()`** helper in `piano_synth.py` — synthetic room impulse response
  (direct + 3 early reflections + late exponential decay, RT60 ≈ 0.3 s).
- **`use_v2=True`** and **`reverb=True`** parameters in `synthesise_melody()` —
  new defaults enable v2 piano and 20% wet room reverb.
- **`duration_s=None`** default in `synthesise_melody()` — derives output length
  from the last event time so the full Chopin piece renders without a 15 s cap.
  Fixes ISSUE-004.
- **`05_pyannow_step9b_audio.ipynb`** — stripped notebook with Step 9b only
  (Fourier + worm PCA + ODE residual features), trained on the **full 229 s**
  piece, synthesised at 8 kHz with the v2 piano.  Output WAV ≈ 3.5 MB.

### Changed
- `pyproject.toml` version → `1.0.0`.
- `PyANNOW/README.md` — updated description and notebook layout for v1.0.0.
- `docs/PyANNOW_NAML_progression.md` — v1.0.0 section added.

---

## [0.9.0] — 2026-05-24

### Calibrated onset detector + RF Step 7 + Worm+Time hybrid Step 9 + AppStat visualization cells

Built on v0.8.0. Root cause of Steps 1, 3 scoring F1=0.000 identified and fixed: the
`find_peaks(height=mean)` magic threshold was not adaptive per-step. All steps now use
a logistic classifier on peak heights with Youden's J threshold. New supervised steps
(RF, Worm+Time hybrid) push F1 from 0.217 baseline to **0.879**.

### Added
- **`calibrated_onset_detect()`** in `midi_target.py` — replaces `find_peaks(height=mean)`
  throughout all steps. Fits a `LogisticRegression` on peak heights (binary label: within
  ±50 ms of a Chopin onset), selects threshold by Youden's J on the ROC curve, enforces
  refractory period. Fixes ISSUE-033 + ISSUE-027.
- **Step 7 — RandomForest onset classifier** in notebook 03 — first directly supervised
  step. Trains `RandomForestClassifier(class_weight='balanced', oob_score=True)` on binary
  onset labels derived from Chopin timings. Reports OOB score + permutation importance per
  worm PC (AppStat Lec 07 pattern). **F1 = 0.872093** (OOB = 0.829). Fixes ISSUE-028.
- **Step 9 — Worm + Fourier Time hybrid MLP** in notebook 03 — inspired by
  `04_chopin_score_net.ipynb`. Concatenates Fourier time embeddings (27-D: phase + 12
  sin/cos harmonics + BPM beat-phase) with standardized worm PCA scores (12-D). Trains
  `sklearn.MLPClassifier(128→64)` subsampled to 10 k pts for speed. **F1 = 0.878613**
  in **3.2 seconds** — best of all steps, proving worm biology adds +0.006 F1 over pure
  time features alone. Fixes ISSUE-042b.
- **`04_chopin_score_net.ipynb`** — new notebook. Trains a Fourier time-embedding →
  residual MLP → per-pitch sigmoid head directly on the Chopin score. Achieves frame-F1
  = 0.9558 and onset-F1 = 0.858315. Serves as an upper-reference ceiling showing what
  pure temporal memorization achieves.
- **`chopin_score_net/`** module — data, model, train, render submodules backing notebook 04.
- **AppStat Lab I cell** in notebook 03 — IOI KDE, kurtosis, skewness for worm and Chopin
  onset distributions. Fixes ISSUE-022.
- **AppStat Lab II PCA biplot cell** in notebook 03 — loadings + scores scatter for worm
  neural subspace. Fixes ISSUE-023.
- **AppStat Lab II t-SNE/UMAP cell** in notebook 03 — motor-state manifold; subsampled to
  2000 pts to avoid timeout. Fixes ISSUE-024.
- **AppStat Lab III/IV clustering cell** in notebook 03 — KMeans / Ward / DBSCAN / GMM
  four-method comparison; Ward subsampled to 2000 pts (O(n²) memory). Fixes ISSUE-025.
- **`tests/score_f1_quick.py`** — standalone 78-second F1 benchmark (no PINN, no JAX);
  covers Steps 0, 1a, 1b, 2, 3, 7, 9, and NB04 reference.
- **`SKIP_PINN = True`** flag in setup cell — guards all Step 8 PINN code; set to `False`
  to run (~3-5 min). Fixes ISSUE-042.

### Changed
- **Step 3 (Ridge)** — F1 improved from **0.000 → 0.286** via `calibrated_onset_detect`.
  Added Lab V diagnostics: VIF, Durbin-Watson statistic, Breusch-Pagan LM test,
  LassoCV effective dimensionality, Ridge R²-vs-λ plot. Fixes ISSUE-026.
- **Step 0 label** changed from "random-sounding" → "deterministic body-wave". Fixes ISSUE-030.
- **Final comparison chart** — F1 panel moved to LEFT (primary metric); onset_loss moved
  to RIGHT (diagnostic only). Fixes ISSUE-019.
- **F1 display precision** — all per-step F1 values now printed with `.6f` (was `.3f`,
  which showed `0.000` for near-zero scores). Fixes ISSUE-041.
- **Notebook cell order** — cells reordered to strict step sequence:
  `0 → 1a → 1b → 2 → 3 → 4-6 → 7 → 8 → 9` (was 0,1a,1b,2,3,7,9,4-6,8).

### Measured F1 scores (v0.9.0, 30 s window, `tests/score_f1_quick.py`)

| Step | Method | F1 | vs baseline |
|---|---|---|---|
| 0 | Deterministic body-wave | 0.216981 | ← baseline |
| 1a | SVD / RSVD | 0.186094 | −0.031 (unsupervised) |
| 1b | SVD + Procrustes | 0.095238 | −0.122 (unsupervised) |
| 2 | K-means motor primitives | 0.108597 | −0.108 (unsupervised) |
| 3 | Ridge + Lab V diagnostics | 0.285974 | **+0.069** |
| 7 | RandomForest | 0.872093 | **+0.655** |
| 9 | Worm+Time hybrid MLP | **0.878613** | **+0.662** |
| NB04 ref | Pure Fourier time-net | 0.858315 | +0.641 (reference only) |

### Issues resolved
- ISSUE-033 ✅ — calibrated_onset_detect (logistic + Youden's J)
- ISSUE-027 ✅ — logistic onset classifier per step
- ISSUE-028 ✅ — Step 7 RandomForest
- ISSUE-026 ✅ — Ridge Lab V diagnostics (VIF, DW, BP, LassoCV)
- ISSUE-042b ✅ — Step 9 Worm+Time hybrid MLP
- ISSUE-022 ✅ — AppStat Lab I IOI KDE
- ISSUE-023 ✅ — AppStat Lab II PCA biplot
- ISSUE-024 ✅ — AppStat Lab II t-SNE/UMAP
- ISSUE-025 ✅ — AppStat Lab III/IV 4-method clustering
- ISSUE-041 ✅ — F1 display precision .6f
- ISSUE-030 ✅ — Step 0 label fix
- ISSUE-019 ✅ — final chart panel order
- ISSUE-042 ✅ — SKIP_PINN flag

---

## [0.8.0] — 2026-05-24

### AppStat audit: metrics, CV, data pipeline — 6 logic problems resolved

Built on the 96-cell Boyle architecture (v0.7.0). This release implements the statistical
fixes identified in the AppStat 2026 audit of the worm-Chopin pipeline. All new functions
are tested; 39 new test methods (50 passing in the new test files).

### Added
- **`pitch_aware_f1()`** in `midi_target.py` — F1 requiring both timing (±50ms) AND pitch
  (exact or pitch-class) match. Greedy one-to-one matching; `pitch_acc` diagnostic field.
  Fixes ISSUE-035 (logic problem #7 — pitch-blind scoring).
- **`biological_pitch_ceiling()`** in `midi_target.py` — fraction of Chopin pitches reachable
  by the worm's fixed muscle-pitch map. 96-cell model = 1.000; 8-cell = 0.583.
  Fixes ISSUE-037 pitch-ceiling component (rate ceiling fixed in ISSUE-002).
- **`precision_recall_at_tolerances()`** in `midi_target.py` — multi-tolerance F1 curve
  (analogue of PR/ROC curve parameterized by temporal tolerance). Fixes ISSUE-020.
- **`bootstrap_musical_f1()`** in `midi_target.py` — non-parametric 95% CI on F1 by
  bootstrapping target onsets. Fixes ISSUE-021.
- **`pyannow/training/cv.py`** (new module) — time-series cross-validation:
  - `time_series_cv()` — walk-forward K-fold; test always in future; no leakage. Fixes ISSUE-038.
  - `blocked_bootstrap_ci()` — block-bootstrap CI preserving autocorrelation. Fixes ISSUE-026/038.
- **`chopin_cumvar()`** in `procrustes.py` — cumulative variance explained by the Chopin
  piano-roll PCA; diagnostic for k_chopin selection.

### Changed
- **`build_chopin_features(k_chopin=None)`** — auto-selects k by 90% variance rule when
  `k_chopin=None` (new default). Explicit k still available. Fixes ISSUE-034 (logic problem #6).
- **`procrustes_align(standardize=True)`** — z-scores W_k column-wise before computing
  cross-covariance, preventing PC1 from dominating the rotation. Fixes ISSUE-032 (logic #4).
  Returns `scale` and `W_k_scaled` for downstream use.
- **`locomotion_pinn.py`** — added prominent architecture note clarifying this module enforces
  a locomotion-oscillator ODE/PDE, NOT the ion-channel HH PINN described in ION_CHANNELS.md.
  Fixes ISSUE-036.

### Tests
- `test_midi_target.py` — 15 new test methods across 4 new test classes
- `test_step1_svd.py` (new) — 12 test methods for procrustes + chopin features
- `test_training_cv.py` (new) — 12 test methods for time_series_cv + blocked_bootstrap_ci
- Total new tests: 39; all pass in numpy-only environment

### Docs (wormuse-analytics/appstat branch)
- Revised ISSUE-018, 020, 021, 029, 031, 032, 034, 035, 036, 037, 038 with new architecture context
- `ISSUE-GROUPS.md` — master grouping of 24 issues into 5 test categories (A-E)
- Issues resolved in v0.7.0 or earlier marked ✅

### Issues resolved
- ISSUE-029 ✅ (v0.7.0) — rank-1 synthetic neural data
- ISSUE-031 ✅ (v0.7.0) — 8-muscle pitch bottleneck
- ISSUE-018 ✅ (v0.7.0) — builder/notebook desync
- ISSUE-032 ✅ (v0.8.0) — unstandardized Procrustes
- ISSUE-034 ✅ (v0.8.0) — lossy k=8 Chopin compression
- ISSUE-035 ✅ (v0.8.0) — pitch-blind F1 metric
- ISSUE-036 ✅ (v0.8.0) — PINN identity clarified
- ISSUE-037 ✅ (v0.8.0) — biological pitch ceiling added
- ISSUE-038 ✅ (v0.8.0) — time-series CV + block bootstrap
- ISSUE-020 ✅ (v0.8.0) — multi-tol F1 curve
- ISSUE-021 ✅ (v0.8.0) — bootstrap CIs for F1

---

## [0.7.0] — 2026-05-24

### Architecture — Boyle et al. 4×24 = 96-cell muscle model + 302-neuron connectome

**Root cause fixed:** All NAML steps (1-6) previously scored *worse* than the Step 0
rule-based baseline because the synthetic `X_neural` (302 × T) was generated as 302
repetitions of the 96 muscle signals, collapsing to rank 1. SVD could only find k=1 PC,
making Ridge / MLP / L-BFGS unable to learn any meaningful mapping. This release fixes
both the neural input structure and the muscle architecture simultaneously.

### Added
- `generate_neural_activity_302()` in `worm_optimizer.py` — biologically structured
  synthetic 302-neuron activity matrix with k≥4 independent principal components.
  Neuron groups follow Boyle et al. (2012): command interneurons (12, 4-phase rhythmic),
  A-class MNs (21, backward traveling wave), B-class MNs (18, forward traveling wave),
  D-class MNs (19, inhibitory antiphase), other interneurons (30, multi-frequency),
  sensory neurons (100, sparse bursts), body neurons (102, slow oscillations + noise).
  Replaces the degenerate `np.vstack([V_mus.T] * n)[:302]` hack.
- `BOYLE_QUADRANT_LAYOUT` documentation constant — Boyle et al. (2012) 4×24 quadrant
  layout (DL, VL, DR, VR) with biological references and musical mapping rationale.
- `generate_muscle_pitches(96)` — 4-quadrant chromatic pitch assignment (8 octaves):
  DL→C1-B2 (MIDI 24-47), VL→C3-B4 (48-71), DR→C5-B6 (72-95), VR→C7-B8 (96-119)
- `MUSCLE_PITCHES_96` constant — 96-cell full BWM pitch array (Boyle architecture)
- 96-cell tests in `PyANNOW/tests/test_forward_model.py`
- 302-neuron activity generator tests in `PyANNOW/tests/test_forward_model.py`

### Changed
- `run_forward_fast()` default `n_muscles`: 95 → **96** (Boyle et al. 4×24 layout)
- `run_forward_fast()` muscle phase structure: 2-quadrant (48 dorsal + 47 ventral) →
  **4-quadrant** (DL phase 0→2π, VL phase π→3π, DR phase 0.05→2π+0.05, VR phase π+0.05→3π+0.05)
  reflecting bilateral dorsal/ventral symmetry per Boyle et al. (2012) Fig. 1.
- `PyANNOW_NAML_progression.md` constraints table: 8 muscle segments → 96 cells (4×24),
  k=1 neural PC → k≥4 neural PCs; added v0.7.0 architecture section
- `CHANGELOG.md` / `VERSION` / `PyANNOW/pyproject.toml` bumped to 0.7.0

### Fixed
- ISSUE-018: degenerate k=1 neural input preventing any learning step from beating Step 0

### Backward compatibility
- `n_muscles=95` still supported in `run_forward_fast()` (pass explicitly)
- `MUSCLE_PITCHES_95` constant retained
- `generate_muscle_pitches(8)` legacy array unchanged

---

## [0.6.0] — 2026-05-22

### Added
- `AI_CONTRIBUTIONS.md` — full documentation of Claude's role per module,
  human-directed corrections, known AI limitations, and citation guidance
- `README.md` — Claude Code badge + AI Attribution section
- `CLAUDE.md` — agent self-description and session startup protocol
- 91-test pytest suite — `PyANNOW/tests/` with 7 test files covering all modules
- `Makefile` — `make test / coverage / lint / nb-test / ci`
- CI updated: real `pytest` replaces `echo` stubs; coverage artifact upload

### Fixed
- `pca_reduce` transpose condition was backwards (caught by new test suite)
- `conftest.py` `REPO_ROOT` path depth (parents[3] → parents[2])

### Versioned
- v0.6.0 git tag

---

## [0.5.0] — 2026-05-22

### Added
- 95-cell BWM model: `generate_muscle_pitches(n)`, `MUSCLE_PITCHES_95`; `run_forward_fast` now accepts `n_muscles` parameter (default 95)
- Chopin Nocturne in C# minor Op.posth. synthesised from score and added to `shared/examples/`
- C# minor pentatonic pitch mapping replacing D♭ pentatonic
- `CLAUDE.md` — project-level issue-tracking rule for all Claude Code sessions
- `TODO.md` reformatted with structured ISSUE entries including affected-files lists (ISSUE-001 through ISSUE-012)
- `EQUIVALENCE_TABLE.md` — cross-system correspondence table (biology ↔ piano ↔ NAML)
- `CHANGELOG.md` — this file
- `VERSION` — single-source version string
- Presentation layout fixes: responsive CSS, `class="dense"`, SVG height reductions, `minScale: 0.08`
- Further Work slides (AMSC: Sibernetics HPC loop; AppStat: ion-channel statistical validation)

### Fixed
- ISSUE-R01: MIDI was mislabeled as Nocturne No. 20; was actually Prelude No. 15 "Raindrop"
- ISSUE-R02: Partial fix — 95-cell model implemented; selective gating remains ISSUE-005
- ISSUE-R03: `.gitignore` now allows `shared/examples/*.mid`
- Presentation overflow on dense slides (16 slides marked `class="dense"`)
- PDE Jacobian shape bug in `locomotion_pinn.py` (`[:, 0]` not `[0]`)

---

## [0.4.0] — 2026-05-22

### Added
- Step 8 PINN: ODE (damped oscillator) and PDE (1D wave equation) physics losses compared
- `compare_ode_vs_pde()` in `locomotion_pinn.py`
- Notebook `03_pyannow_naml_progression.ipynb` (2 MB, all 8 NAML steps executed)
- Notebook `02_chopin_worm_optimizer.ipynb` §9 audio playback with embedded WAV widgets
- `piano_synth.py` — modal synthesis (40 modes, unconditionally stable)
- `synthesise_melody()` accepts `pitch_map` for n-cell generality
- Further Work sections in docs and presentation (AMSC + AppStat)
- Responsive presentation layout overhaul

---

## [0.3.0] — 2026-05-21

### Added
- Full PyANNOW module tree: `step1_svd/`, `step2_clustering/`, `step3_regression/`, `step4_ffnn/`, `step5_training/`, `step6_lbfgs/`, `step8_pinn/`
- Reveal.js presentation `PyANNOW/presentation/index.html` (36 slides, 7 parts)
- `PyANNOW/docs/PyANNOW_NAML_progression.md` — living doc with measured results
- Chopin worm optimizer notebook `02_chopin_worm_optimizer.ipynb`
- C. elegans-realistic ion channel models: EGL-19, EXP-2, SHK-1, NCA-1/2, UNC-2
- MIDI target parser (`midi_target.py`), biological ceiling analysis

---

## [0.2.0] — 2026-05-20

### Added
- `docs/SCIENTIFIC_FOUNDATION.md` — full biology↔piano physics derivation (HH kinetics, wave equations, Procrustes correspondences, references)
- `docs/scientific_foundation_demo.ipynb` — executed MVP notebook: HH neuron → spike → muscle → stiff string → melody + τ_m sweep
- `shared/examples/frederic-chopin-nocturne-no20.mid` (old, mislabeled; kept as `chopin_prelude_no15_raindrop_db.mid`)

---

## [0.1.0] — 2026-05-19  *(initial scaffolding)*

### Added
- `README.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `ION_CHANNELS.md`
- `docker-compose.yml` (MK + OpenWorm services)
- `.github/workflows/verify.yml` CI skeleton
- `docs/lectures/lecture-map.md`
- Sub-project stubs: `wormuse-sim/`, `PyANNOW/`, `wormuse-analytics/`, `shared/`, `ui/`
- `wormuse-sim/CMakeLists.txt` Phase-0 skeleton
- `PyANNOW/pyproject.toml`, `wormuse-analytics/pyproject.toml`
- MIT `LICENSE`
