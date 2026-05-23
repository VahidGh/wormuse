# AI Contributions

This repository was built with substantial assistance from **Claude** (claude-sonnet-4-5 / claude-sonnet-4-6),
Anthropic's AI assistant, running inside **Claude Code** — Anthropic's agentic
coding tool for the terminal (and VS Code extension).

---

## Collaboration model

| Role | Person / System |
|---|---|
| **Project concept & scientific ideas** | Vahid Ghayoomie |
| **Biological grounding & correctness** | Vahid Ghayoomie |
| **Issue discovery, validation & direction** | Vahid Ghayoomie |
| **All implementation (code, docs, notebooks, tests)** | Claude |
| **Issue tracking, architecture decisions** | Claude (under Vahid's direction) |

**Vahid Ghayoomie** (Politecnico di Milano, 2025-26) is the researcher and
scientific director. He conceived the project, defined all its scientific ideas
and goals, discovered and logged every issue in `TODO.md`, and validated
results at each step — listening to audio output, checking plots, reviewing
scientific correctness, and deciding what gets built next.

**Claude** acted as principal development engineer: writing all code,
documentation, tests, and the presentation, under Vahid's direction.

---

## What Vahid contributed

### The concept and scientific framework

- **"Can a worm play Chopin?"** — the original question, the biological model (*C. elegans*), and the musical target (Chopin Nocturne No. 20 in C# minor)
- **NAML course as the narrative spine** — the idea of using each lecture's method (SVD → PCA → ridge → MLP → L-BFGS → PINN) as a progressive development story
- **Biological grounding** — deciding which ion channels to model, what biological constraints matter, and when the model violated biological realism

### Issue discovery and validation (full TODO.md history)

Every item in `TODO.md` was either raised by Vahid during review or emerged
directly from his validation of Claude's output:

| Issue | What Vahid caught |
|---|---|
| **ISSUE-001** | "The note is from Chopin Nocturne 20, you mentioned 15" — discovered the MIDI mislabeling across code, notebooks, docs, and presentation |
| **ISSUE-002** | Identified that the `biological_ceiling()` formula was scientifically wrong (single-voice greedy ≠ n-voice capacity); directed the fix |
| **ISSUE-003** | Judged the modal synthesis piano as sounding like a "tin can" — requested realistic audio |
| **ISSUE-004** | Noted the piece was capped at 15 s / 40 notes; requested full-length render |
| **ISSUE-005** | Identified that the 95-cell model fires 38 notes/s — biologically too dense — as a design issue requiring selective gating |
| **ISSUE-006** | Proposed dorsal/ventral antiphase exploitation as a biological realism improvement (2× note density) |
| **ISSUE-007** | Caught the notebook `n_muscles` hardcoding inconsistency |
| **ISSUE-008** | Proposed Ca_thresh as a PINN-tunable parameter (connecting to SC-PINN project) |
| **ISSUE-009 – 015** | All initiated or reviewed by Vahid |

### Feature direction

- Requested audio players in notebooks (synthesised worm melody, then real piano WAV)
- Directed replacement of the failed `html-midi-player` approach with WAV synthesis
- Requested background Chopin soundtrack in the presentation with play/pause controls
- Confirmed ODE + PDE PINN split at Step 8; approved Procrustes at Step 1
- Set all constraints on scope ("use only NAML course materials", "not PyTorch")

### Corrections that changed the implementation

| Correction | Outcome |
|---|---|
| "Use only NAML course materials" | Constrained all ML methods to specific lectures/labs |
| "The canonical Python stack is from the lab notebooks" | Switched from PyTorch recommendation to the actual course stack (JAX/Flax/Optax) |
| "There is nothing regarding Python ML libraries needed inside this container" | Stopped Claude probing the MK docker for Python packages |
| "Use `source /u/sw/etc/profile.d/mk.sh`" | Corrected Claude's use of `bash.bashrc` which bails on non-interactive shells |
| Noted rate mismatch (95-cell fires 38/s) | Logged as ISSUE-005; led to selective gating research direction |
| "The note is from Chopin Nocturne 20, you mentioned 15" | Triggered full ISSUE-001 audit across all files |
| Biological ceiling formula wrong | Triggered ISSUE-002; corrected from 57.7% (single-voice artefact) to ~100% |

---

## What Claude built (module by module)

### Scaffolding & infrastructure
- `README.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `ION_CHANNELS.md` — written by Claude
- `docker-compose.yml`, `.github/workflows/verify.yml`, `Makefile` — designed by Claude
- `CLAUDE.md`, `TODO.md`, `CHANGELOG.md`, `VERSION` — written by Claude

### PyANNOW Python library
- `ion_channels/celegans_hh.py` — C. elegans channel models (EGL-19, EXP-2, SHK-1, NCA, UNC-2); verified against Jospin et al. 2002
- `targets/midi_target.py` — MIDI parsing, onset_loss, biological_ceiling (n-voice)
- `composer/worm_optimizer_fast.py` — vectorised 95-cell forward model
- `composer/piano_synth.py` — modal synthesis synthesiser (FDM version replaced after Claude diagnosed CFL instability)
- `composer/worm_optimizer.py` — `generate_muscle_pitches(n)`
- Steps 1–8 (`step1_svd/` through `step8_pinn/`) — written by Claude per Vahid's NAML lecture mapping
- `step8_pinn/locomotion_pinn.py` — ODE and PDE PINN versions (PDE Jacobian bug fixed after shape error)

### Test suite
- All 7 test files (91 tests) — written by Claude
- Bug caught: `pca_reduce` backwards transpose condition — identified via test failure

### Documentation
- `docs/SCIENTIFIC_FOUNDATION.md`, `docs/EQUIVALENCE_TABLE.md` — written by Claude
- `PyANNOW/docs/PyANNOW_NAML_progression.md` — maintained by Claude
- `CHANGELOG.md` — compiled by Claude

### Presentation
- `PyANNOW/presentation/index.html` — 36-slide Reveal.js presentation (CSS, SVG, content, audio controls)

### Notebooks
- `PyANNOW/notebooks/02_chopin_worm_optimizer.ipynb` — built and executed by Claude
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — built and executed by Claude

---

## Known AI limitations encountered

1. **MIDI sourcing:** Claude could not download the correct Chopin MIDI from the internet. Vahid sourced and provided the correct file.

2. **PDE Jacobian bug:** The `pde_residual` function used `[0]` instead of `[:, 0]` for column-slicing. Fixed after shape error appeared during execution.

3. **`pca_reduce` transpose bug:** The condition `X.shape[0] > X.shape[1]` was backwards. Caught by the test suite.

4. **FDM instability:** Initial piano string model violated the CFL stability condition for high strings. Replaced with modal synthesis.

5. **Optimisation sensitivity:** Nelder-Mead showed near-zero improvement because the loss is insensitive to ion-channel parameters in the regular-wave model. Claude documented this rather than hiding it; Vahid confirmed this reflected a real biological finding.

6. **Single-voice ceiling bug:** `biological_ceiling()` used a single greedy voice, yielding 57.7% — a mathematical artefact, not biology. Vahid identified it as wrong; Claude computed the correct n-voice result (~100%).

---

## How to reproduce this workflow

```bash
npm install -g @anthropic-ai/claude-code
cd /Users/vghayoomie/git/wormuse
claude
```

The `polimi-naml` skill (in `~/.claude/skills/polimi-naml/`) was the primary
skill used for PyANNOW. It includes the course lecture map, JAX/Flax/Optax
stack, and PINN reference from `naml-ion-channel-pinn/`.

---

## Suggested citation

```bibtex
@misc{wormuse2026,
  author       = {Ghayoomie, Vahid},
  title        = {wormuse: A \textit{C. elegans}--driven musical simulator},
  year         = {2026},
  note         = {Built with Claude Code (Anthropic). Version 0.6.0},
  howpublished = {\url{https://github.com/[username]/wormuse}}
}
```

---

## Claude versions used

| Period | Model |
|---|---|
| Architecture, scaffolding, AMSC/NLA/NMPDE skills | claude-sonnet-4-5 |
| PyANNOW modules, notebooks, test suite, presentation | claude-sonnet-4-6 |

All interactions were via Claude Code in the macOS terminal and VS Code extension.
