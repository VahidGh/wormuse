# Architecture

This document describes the modules of **wormuse**, how they communicate, and where data lives. For the *physics* of why the worm-to-piano pipeline is coherent (not just convenient), read [docs/SCIENTIFIC_FOUNDATION.md](./docs/SCIENTIFIC_FOUNDATION.md) — the implementation files in the sub-projects reference equation numbers from that document (e.g. "implements equation A.2 with the θ-method from B.3").

---

## Modules

### `wormuse-sim/` — AMSC project (C++ simulator core)

| Component | Responsibility |
|---|---|
| `src/ow_bridge/` | Spawns the OpenWorm Docker container, drives Sibernetic + C302, captures stdout / WCON / spike events. Two backends: `docker_runner` (CLI subprocess) and `lib_bindings` (future: C-ABI wrapper). |
| `src/piano_sim/` | Modern C++ piano. Phase 2 = 1D wave equations per string with OpenMP. Phase 5 = full hammer–string–soundboard FEM in deal.II (reuses NMPDE Heat.cpp infrastructure). |
| `src/neuron_to_midi/` | Translates C302 spike events → MIDI / piano-trigger schedule. Pluggable mapping policies (one-neuron-per-key, motor-primitive-to-chord, ion-channel-tuned). |
| `src/orchestrator/` | The main timeline: integrates the worm body, the piano, and the I/O. Writes outputs to `shared/data_formats/`. |
| `apps/wormuse_sim.cpp` | CLI driver. |
| `tests/` | CTest unit tests. |

**Build environment:** MK Docker (`quay.io/pjbaioni/amsc_mk:2025` with `gcc-glibc/11.2.0 + dealii + eigen + lis`).

### `PyANNOW/` — NAML project (composer)

| Component | Responsibility |
|---|---|
| `src/pyannow/ion_channels/` | **The PINN core.** JAX implementation of Hodgkin-Huxley neuron + ion-channel kinetics (EGL-19, EXP-2, NCA-1/2, SHK-1, UNC-2). |
| `src/pyannow/step1_svd/` | RSVD encoder + Procrustes alignment. Compresses 302-D neural activity → k PCs; rotates into Chopin feature space. |
| `src/pyannow/step2_clustering/` | PCA + K-means motor primitives. Finds discrete behavioural states in the neural trajectory. |
| `src/pyannow/step3_regression/` | Ridge regression composer + Lab V diagnostics (VIF, Durbin-Watson, Breusch-Pagan, LassoCV). |
| `src/pyannow/step4_ffnn/` | JAX/Flax MLP composer (k_worm → hidden → k_chopin). |
| `src/pyannow/step5_training/` | Adam mini-batch trainer (course-canonical from NAML Lab 10). |
| `src/pyannow/step6_lbfgs/` | L-BFGS polish stage (NAML L22 / Lab08). |
| `src/pyannow/step8_pinn/` | ODE + PDE PINN locomotion constraints (damped oscillator + 1D wave equation). |
| `src/pyannow/targets/midi_target.py` | MIDI parser, onset metrics (musical_f1, ioi_similarity, onset_loss), calibrated_onset_detect (logistic + Youden's J). |
| `src/pyannow/training/cv.py` | Time-series CV (walk-forward) + blocked bootstrap CI. |
| `notebooks/03_pyannow_naml_progression.ipynb` | **Main NAML notebook** — 9-step progression (SVD → PCA → Ridge → MLP+Adam+LBFGS → RF → PINN → Worm+Time hybrid). Measured F1: Step 0=0.217 → Step 9=0.879. |
| `notebooks/04_chopin_score_net.ipynb` | Fourier time-embedding → residual MLP → piano-roll reconstruction. Reference ceiling F1=0.858 on pure time features. |
| `chopin_score_net/` | Module backing notebook 04: data, model, train, render submodules. |

**Environment:** local Python 3.10-3.13, JAX/Flax/Optax stack matching the NAML labs (see `pyproject.toml`).

### `wormuse-analytics/` — AppStat project (statistical analysis)

| Component | Responsibility |
|---|---|
| `notebooks/Lab_I_descriptive.ipynb` | Descriptive stats on worm motion + spike rates. Kurtosis, skew, distributions per neuron. |
| `notebooks/Lab_II_PCA.ipynb` | PCA on 302-D neural state trajectories. Scree plot, biplot, choose k. |
| `notebooks/Lab_III_clustering.ipynb` | Cluster motor primitives (forward / backward / Ω-turn) from PCA-reduced trajectories. KMeans + Ward + silhouette. |
| `notebooks/Lab_V_regression.ipynb` | OLS regression of a "music quality" scalar on ion-channel parameters. Full diagnostics (BP / DW / VIF / Cook's). |
| `notebooks/Lab_VI_classification.ipynb` | Binary classifier: "musical" vs "noisy" generated melodies. Logistic + ROC + RF baseline. |
| `src/wormuse_analytics/` | Shared helpers: metric implementations, plotting style, data loaders. |

**Environment:** local Python with numpy / pandas / scikit-learn / statsmodels / matplotlib / seaborn / umap (mirrors AppStat 2026 labs).

### `shared/` — bridge between sub-projects

| Sub-folder | Contents |
|---|---|
| `data_formats/` | JSON / WCON / MIDI schemas, version-bumped. The contract between C++ and Python. |
| `parameter_schemas/` | Pydantic models for ion-channel parameters, composer hyperparameters, piano configuration. |
| `examples/` | Pre-computed reference scenarios (small WCON traces + spike events + golden audio) used by tests and the UI. |

### `ui/` — visualization layer

| Sub-folder | Contents |
|---|---|
| `notebook/` | A single JupyterLite-compatible notebook. Runs in the browser via Pyodide. Lets the user tweak a few PINN parameters live and hear the melody morph (uses a pre-trained surrogate, no heavy simulator in the browser). |
| `static/` | Pure HTML+JS site for GitHub Pages. Three.js worm-body animation + Web Audio API piano. Pre-rendered scenarios. **Primary GH Pages deliverable.** |
| `render/` | Offline render scripts that produce the JSON / MIDI assets shipped to `static/data/`. Run on a developer machine with the full simulator. |

### `docs/`

| Sub-folder | Contents |
|---|---|
| `lectures/lecture-map.md` | Master table: each code module ↔ the lecture(s) it implements. |
| `design_notes/` | Design rationale for non-obvious choices (e.g., why JAX over PyTorch, why deal.II for soundboard). |
| `math_derivations/` | Hand-typed derivations of weak forms, error estimates, PINN loss decompositions. |
| `figures/` | Diagrams, plots, screenshots. |

---

## Data flow (single run)

```
1. user → docker compose up wormuse-sim
2. wormuse-sim spawns OpenWorm container
3. OpenWorm runs N seconds of Sibernetic + C302
4. wormuse-sim collects: pose (WCON), spikes (JSON), muscle (CSV)
5. neuron_to_midi/ translates spikes → MIDI events (subject to ion-channel timing)
6. piano_sim/ ingests MIDI → produces audio (WAV)
7. orchestrator/ writes one scenario to shared/examples/run_NNN/
   ├── pose.wcon
   ├── spikes.json
   ├── midi.mid
   └── audio.wav

8. PyANNOW reads scenarios → trains composer + PINN
9. wormuse-analytics reads scenarios + composer outputs → notebooks
10. ui/render/ packs subset of scenarios → JSON for static UI
```

## Build / runtime topology

| Tool | Where it runs | Why |
|---|---|---|
| Sibernetic (OpenCL-GPU SPH) | OpenWorm Docker container | Pre-built by the OpenWorm team; CL-only |
| C302 (Python + NEURON) | OpenWorm Docker container | Bundled with Sibernetic in the same image |
| deal.II piano FEM | MK Docker container | We need `gcc-glibc/11.2.0 + dealii + lis` |
| PyANNOW (JAX) | Host Python or a sibling container | JAX install user-managed; GPU optional |
| wormuse-analytics | Host Python or JupyterLab container | Lightweight numpy / sklearn |
| UI (static) | Browser, hosted on GitHub Pages | Pure HTML + JS |
| UI (JupyterLite) | Browser, hosted on GitHub Pages | Pyodide for live tweaks |

A `docker-compose.yml` at the repo root composes the MK and OpenWorm containers and mounts `shared/` so all three sub-projects share data.

## Why no external dependencies on the user's other repos

Per design decision: this repo is **self-contained**. The PINN ideas from `naml-ion-channel-pinn/` and the piano-physics ideas from polimuse are **re-implemented inline** in `PyANNOW/src/pyannow/ion_channels/` and `wormuse-sim/src/piano_sim/`, respectively. This keeps the project portable (anyone can clone and build without chasing sibling repos) at the cost of some code duplication. References / credit go into `docs/`.

## Performance budget (rough)

| Operation | Target time (laptop) |
|---|---|
| 60-second worm sim (Sibernetic) | 5-15 minutes (GPU-accelerated OpenCL) |
| 60-second C302 spike train | 1-3 minutes |
| 60-second piano FEM (deal.II, P¹, h=2cm) | 30-60 seconds |
| One PINN training run (toy HH) | 5-10 minutes |
| Full scenario render → MP3 + JSON | ~20 minutes end-to-end |

The UI ships pre-rendered scenarios so end users see results instantly.
