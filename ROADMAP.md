# Roadmap

8 phases, ~8-10 weeks of focused work alongside the courses.

---

## Phase 0 — Foundation ✅

**Goal:** repo scaffolding, design docs, the development environment specified.

- [x] `git init` at `/Users/vghayoomie/git/wormuse`
- [x] MIT LICENSE
- [x] `README.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `ION_CHANNELS.md`
- [x] Three sub-project READMEs with lecture maps
- [x] `docker-compose.yml` (MK + OpenWorm)
- [x] `.github/workflows/verify.yml` skeleton
- [x] `docs/lectures/lecture-map.md` master table
- [x] `.gitignore`

**Exit criteria:** `git log` shows the initial commit; running `tree` shows the full structure.

---

## Phase 1 — Bridge layer (3-5 days)

**Goal:** prove we can drive OpenWorm and read its output reliably.

- [ ] `scripts/run_openworm.sh` — driver that runs Sibernetic + C302 for N seconds inside the container with a fixed config
- [ ] `wormuse-sim/src/ow_bridge/docker_runner.{hpp,cpp}` — minimal C++ wrapper that calls `docker run`, waits, captures outputs
- [ ] `shared/data_formats/spike_event.json` — JSON schema (timestamp, neuron_id, voltage, source)
- [ ] `shared/data_formats/worm_pose.md` — explanation of WCON used by OpenWorm
- [ ] Python parser in `wormuse-analytics/src/wormuse_analytics/loaders.py` that reads spike + WCON files
- [ ] **Smoke test:** 30 second worm run → events file → Python notebook plots spike raster ✅

**Exit criteria:** one end-to-end run from `make run` to a saved spike raster PNG.

---

## Phase 2 — Piano MVP (1 week)

**Goal:** simplest piano model that converts spike events to audio.

- [ ] `wormuse-sim/src/piano_sim/string_1d.{hpp,cpp}` — 1D wave equation per string, finite differences, modal decomposition
- [ ] Eight strings (C-major octave) hard-coded
- [ ] OpenMP `#pragma omp parallel for` over strings
- [ ] `wormuse-sim/src/neuron_to_midi/policy_one_to_one.{hpp,cpp}` — map first 8 motor neurons to the 8 strings
- [ ] Output a WAV file via libsndfile or a custom RIFF writer
- [ ] **Demo:** worm crawl → piano scale played in sequence ✅

**Exit criteria:** `wormuse_sim --input run000/spikes.json --output run000/audio.wav` produces audible WAV.

---

## Phase 3 — PyANNOW PINN spike (1 week) ✅ (core done, v0.8.0)

**Goal:** PINN tuned ion-channel module that emits realistic firing schedules.

- [x] `PyANNOW/src/pyannow/ion_channels/celegans_hh.py` — Hodgkin-Huxley ODE in JAX (pure functional, 6-channel: EGL-19, EXP-2, NCA-1/2, SHK-1, UNC-2)
- [x] `PyANNOW/src/pyannow/step8_pinn/locomotion_pinn.py` — ODE + PDE PINN with physics-residual loss
- [x] Adam → L-BFGS training loop (course-canonical from NAML L22 + L27)
- [x] `notebooks/03_pyannow_naml_progression.ipynb` Step 8 — PINN inside the full progression (SKIP_PINN=True by default, ~5 min if enabled)
- [ ] `notebooks/01_ion_channels_pinn.ipynb` — dedicated PINN training notebook (stretch)
- [x] **Demo:** Step 8 PINN reproduces locomotion ODE/PDE residual; guarded by `SKIP_PINN` flag ✅

**Exit criteria:** notebook end-to-end with convergence plot and held-out validation.

---

## Phase 4 — Analytics notebooks (1 week) ✅ (PyANNOW portion live, v0.9.0)

**Goal:** the five AppStat-style notebooks with the worm dataset.

- [ ] Run the bridge layer 50 times with varied ion-channel parameters (blocked on Phase 1)
- [ ] Collect into `shared/examples/dataset_v1/` (≥ 50 scenarios)
- [x] AppStat Lab I–IV visualizations embedded in `notebooks/03_pyannow_naml_progression.ipynb` (IOI KDE, PCA biplot, t-SNE/UMAP, 4-method clustering)
- [x] AppStat Lab V Ridge diagnostics (VIF, Durbin-Watson, Breusch-Pagan, LassoCV) in Step 3
- [x] AppStat Lab VI classification: Step 7 RandomForest (F1=0.872, OOB=0.829) + Step 9 MLP (F1=0.879)
- [x] `notebooks/04_chopin_score_net.ipynb` — Fourier time-embedding → residual MLP reference ceiling (F1=0.858)
- [ ] `wormuse-analytics/notebooks/` standalone analytics notebooks (blocked on bridge dataset)

**Exit criteria:** each notebook runs end-to-end with markdown interpretations.

---

## Phase 5 — Piano FEM upgrade (2 weeks)

**Goal:** real physical piano via deal.II soundboard FEM (reuses NMPDE infrastructure).

- [ ] `wormuse-sim/src/piano_sim/soundboard_fem.{hpp,cpp}` — vibrating plate equation, Q1 elements
- [ ] Adapts the user's `nmpde-projects/src/Heat.cpp` CI infrastructure (CMake, testing, scaling)
- [ ] String → soundboard coupling at the bridge nodes
- [ ] Hammer model: nonlinear contact force with felt
- [ ] **Demo:** single piano key strike produces realistic decay envelope + harmonic series ✅

**Exit criteria:** comparison plot vs reference Chabassier piano model (in `polimuse-docs/Time Domain Simulation of a Piano Part 2`).

---

## Phase 6 — End-to-end pipeline (1 week)

**Goal:** one scenario flows: worm → spikes → composer → MIDI → piano → WAV.

- [ ] `PyANNOW/src/pyannow/composer/seq2seq.py` — neural state latent → MIDI sequence (Flax)
- [ ] Trained on Phase 4 dataset, with the PINN-tuned spikes as condition
- [ ] `scripts/render_scenario.py` — full pipeline runner
- [ ] **Demo:** picking three ion-channel parameter sets produces three audibly distinct melodies, all musically plausible ✅

**Exit criteria:** three scenario WAVs + their JSON metadata in `shared/examples/dataset_v2/`.

---

## Phase 7 — UI (1-2 weeks)

**Goal:** GitHub-Pages-deployable interactive demo.

- [ ] `ui/render/build_static_dataset.py` — packs ≥ 10 scenarios into JSON + MIDI assets
- [ ] `ui/static/index.html` — Three.js worm + piano + Web Audio playback
- [ ] Sliders: scenario picker, playback control, single-neuron focus
- [ ] `ui/notebook/wormuse.ipynb` — JupyterLite-compatible notebook with ipywidgets
- [ ] `.github/workflows/gh-pages.yml` — auto-deploy on push to `main`

**Exit criteria:** GH Pages URL plays a worm-driven melody in any modern browser, no install required.

---

## Phase 8 — Polish + docs (ongoing)

- [ ] Complete `docs/lectures/lecture-map.md` (one row per lecture across all 7 courses)
- [ ] Per-module `docs/design_notes/*.md` for non-obvious choices
- [ ] `docs/math_derivations/*.md` for the piano wave equation, PINN loss, etc.
- [ ] 60-second demo video / GIF for the README
- [ ] (Stretch) Multi-cultural melody generation per polimuse's original ambition
- [ ] (Stretch) Wire in the user's full SC-PINN as an alternative ion-channel backend

---

## Cross-course mapping

| Course | Phases | Where |
|---|---|---|
| AMSC | 1, 2, 5, 6 | `wormuse-sim/` throughout |
| NAML | 3, 6 | `PyANNOW/` ion-channel + composer |
| AppStat | 4 | `wormuse-analytics/` notebooks |
| NMPDE | 5 | piano FEM (reuses Heat.cpp patterns) |
| NLA | 5 | sparse solvers inside the FEM |
| PC | 2, 5 | OpenMP / MPI in piano + Sibernetic |
| SE4HPC | 0, 7 | CI/CD + verification + scalability |

## Pacing notes

- Phases 1-3 can overlap (different sub-projects).
- Phase 5 (deal.II) is the longest single phase — start early after Phase 2.
- The UI (Phase 7) is mostly independent and can be sketched in parallel with Phases 3-5 using stub data.
