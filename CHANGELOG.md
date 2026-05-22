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

## [0.6.0] — 2026-05-22  *(current)*

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
