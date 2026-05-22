# AI Contributions

This repository was built with substantial assistance from **Claude** (claude-sonnet-4-5 / claude-sonnet-4-6),
Anthropic's AI assistant, running inside **Claude Code** — Anthropic's agentic
coding tool for the terminal (and VS Code extension).

---

## Role of the AI agent

Claude acted as the **principal development engineer** throughout the project,
working under direct guidance and design decisions from the human researcher
(Vahid Ghayoomie, Politecnico di Milano, 2025-26).

The collaboration model was:
- **Human:** provided scientific direction, course material context,
  confirmed design choices, caught domain-specific errors, set constraints
- **Claude:** implemented code, wrote documentation, designed architecture,
  scaffolded tests, identified bugs, built the presentation, and maintained
  the issue tracker

---

## What Claude built (module by module)

### Scaffolding & infrastructure (`Phase 0`)
- `README.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `ION_CHANNELS.md` — written by Claude
- `docker-compose.yml` — designed by Claude
- `.github/workflows/verify.yml` — initially scaffolded by Claude, updated to real pytest
- `CLAUDE.md`, `TODO.md`, `CHANGELOG.md`, `VERSION` — written by Claude
- `Makefile` — written by Claude

### PyANNOW Python library
- `ion_channels/celegans_hh.py` — C. elegans channel models (EGL-19, EXP-2, SHK-1, NCA, UNC-2) written by Claude, verified against Jospin et al. 2002 literature
- `targets/midi_target.py` — MIDI parsing, onset_loss, biological_ceiling — written by Claude
- `composer/worm_optimizer_fast.py` — vectorised 95-cell forward model — written by Claude (noted and corrected by human for the 95-cell rate issue)
- `composer/piano_synth.py` — modal synthesis synthesiser — written by Claude (FDM version replaced with modal after Claude diagnosed CFL instability)
- `composer/worm_optimizer.py` — `generate_muscle_pitches(n)` — written by Claude
- Steps 1-8 (`step1_svd/` through `step8_pinn/`) — written by Claude per NAML lecture mapping
- `step8_pinn/locomotion_pinn.py` — both ODE and PDE PINN versions — written by Claude (PDE Jacobian bug later caught by tests)

### PyANNOW test suite
- All 7 test files (`tests/conftest.py` + 6 test modules, 91 tests) — written by Claude
- Bug caught by tests: `pca_reduce` backwards transpose condition — identified by Claude via test failure

### Documentation
- `docs/SCIENTIFIC_FOUNDATION.md` — physics derivations (HH kinetics through piano PDE); written by Claude, equations derived by Claude from literature
- `docs/EQUIVALENCE_TABLE.md` — 20-row cross-system correspondence table — designed and written by Claude
- `PyANNOW/docs/PyANNOW_NAML_progression.md` — living document — maintained by Claude
- `CHANGELOG.md` — retroactive version history — compiled by Claude

### Presentation
- `PyANNOW/presentation/index.html` — 36-slide Reveal.js presentation — designed and written by Claude (CSS, SVG animations, slide content)

### Notebooks
- `docs/scientific_foundation_demo.ipynb` — fully executed MVP notebook — built by Claude using a Python builder script
- `PyANNOW/notebooks/02_chopin_worm_optimizer.ipynb` — built and executed by Claude
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — built and executed by Claude

---

## Where the human directed / corrected Claude

These are examples where the human's scientific judgment was essential:

| Decision / correction | Outcome |
|---|---|
| "Use only NAML course materials" | Constrained all ML methods to specific lectures/labs |
| "The canonical Python stack is from the lab notebooks, not PINN requirements" | Switched JAX recommendation from PyTorch to the actual course stack |
| "There is nothing regarding Python ML libraries needed inside this container" | Stopped Claude probing the MK docker for Python packages |
| "Use `source /u/sw/etc/profile.d/mk.sh`" | Corrected Claude's use of `bash.bashrc` which bails on non-interactive shells |
| Confirmed ODE+PDE comparison in Step 8 | Claude proposed it; human confirmed the split |
| Noted rate mismatch (95-cell fires 38/s) | Claude's forward model was correct; human flagged it as a design issue (ISSUE-005) |
| "The note is from Chopin Nocturne 20, you mentioned 15" | Prompted Claude to investigate and discover the MIDI mislabeling (ISSUE-R01) |
| Approved Procrustes at Step 1 and PINN at Step 8 | Human chose between Claude's proposed alternatives |

---

## Known AI limitations encountered

1. **MIDI sourcing:** Claude could not download the correct Chopin MIDI from the internet (piano-midi.de was unreachable; bitmidi.com returned Final Fantasy II). Synthesised the piece from score as a workaround. A human with browser access could download a higher-quality MIDI.

2. **PDE Jacobian bug:** The `pde_residual` function used `[0]` instead of `[:, 0]` for column-slicing the Jacobian matrix. Claude fixed it after the shape error appeared during execution.

3. **`pca_reduce` transpose bug:** The condition `X.shape[0] > X.shape[1]` was backwards — never transposed for typical 302×500 neural activity. Caught by the test suite Claude wrote for itself.

4. **FDM instability:** The initial piano string model used finite-difference methods that violated the CFL stability condition for high strings. Claude diagnosed and replaced with modal synthesis.

5. **Optimization sensitivity:** The Nelder-Mead optimizer for the Chopin workflow showed near-zero improvement (0%) because the loss function is insensitive to ion-channel parameters in the regular-wave model. This reflects a real biological finding (timing is structural, not parametric) — Claude documented this rather than hiding it.

---

## How to reproduce this workflow

The entire project was built using:
```bash
# 1. Install Claude Code
npm install -g @anthropic-ai/claude-code

# 2. Navigate to project root
cd /Users/vghayoomie/git/wormuse

# 3. Start a session (Claude reads CLAUDE.md automatically)
claude

# 4. Project-specific skills were loaded from:
#    ~/.claude/skills/polimi-{amsc,naml,nla,nmpde,pc,appstat,appstat-r}/
```

The `polimi-naml` skill (in `~/.claude/skills/polimi-naml/`) was the primary
skill used for PyANNOW. It includes:
- Course lecture map (27 lectures → code locations)
- JAX/Flax/Optax canonical stack from lab imports
- PINN reference from `naml-ion-channel-pinn/`
- Code snippets for SVD, RSVD, PCA, optimisation, PINN

---

## Suggested citation

If citing this work, include:
```
wormuse: A C. elegans-driven musical simulator
Vahid Ghayoomie, Politecnico di Milano, 2026
Built with Claude Code (Anthropic)
https://github.com/[username]/wormuse
```

Or in BibTeX:
```bibtex
@misc{wormuse2026,
  author       = {Ghayoomie, Vahid},
  title        = {wormuse: A \textit{C. elegans}–driven musical simulator},
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
| PyANNOW modules, notebooks, test suite, wormuse build | claude-sonnet-4-6 |

All interactions were via Claude Code in the macOS terminal (with occasional VS Code).
