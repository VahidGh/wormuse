# wormuse — Claude Code project instructions

## Issue tracking rule (ALWAYS follow)

Every time you identify a bug, missing feature, incorrect assumption, or improvement opportunity:

1. **Add it to `TODO.md` immediately** — before finishing the conversation turn.
2. **Include an "Affected files" list** for every entry — explicitly name:
   - Source files (`.py`, `.cpp`, `.hpp`)
   - Documentation (`.md` files, living docs)
   - Notebooks (`.ipynb`)
   - Presentations (`presentation/index.html`)
   - Build scripts (`_build_*.py`)
   - Any generated outputs that will need re-running
3. **Mark resolved** — when an issue is fixed, update its status in TODO.md in the same commit.
4. **Link to commit** — add the fixing commit hash to the TODO entry when resolved.

### TODO.md entry format

```markdown
### ISSUE-NNN — Short descriptive title

| Field | Value |
|---|---|
| **Status** | 🔴 Open / 🟡 In Progress / ✅ Resolved (commit abc1234) |
| **Priority** | P0 / P1 / P2 / P3 |
| **Severity** | Bug / Improvement / Correctness / Performance |

**Description:** One paragraph explaining what's wrong and why.

**Affected files:**
- `path/to/file.py` — what needs changing
- `PyANNOW/notebooks/02_chopin_worm_optimizer.ipynb` — needs re-execution
- `PyANNOW/presentation/index.html` — slide text update
- `TODO.md` — this entry

**Fix plan:** Concrete steps to resolve it.
```

## Repository overview

See `README.md`, `ARCHITECTURE.md`, `ROADMAP.md`, and `ION_CHANNELS.md` for the
project description. The active work log lives in `PyANNOW/docs/PyANNOW_NAML_progression.md`.

## Key paths

| Path | Purpose |
|---|---|
| `shared/examples/*.mid` | MIDI targets (Nocturne C# minor, Raindrop Prelude) |
| `PyANNOW/src/pyannow/` | All Python modules |
| `PyANNOW/notebooks/` | Executed Jupyter notebooks |
| `PyANNOW/presentation/index.html` | Reveal.js presentation |
| `PyANNOW/docs/` | Living documentation |
| `wormuse-sim/` | C++ simulator (Phase 1+ of ROADMAP) |
| `wormuse-analytics/` | AppStat-style notebooks |

## Toolchain

- Python: local miniconda env (Python 3.13)
- C++: `quay.io/pjbaioni/amsc_mk:2025` docker with `module load gcc-glibc/11.2.0 dealii lis`
- OpenWorm: `openworm/openworm:latest` docker
- Packages: `pip install -e PyANNOW` (editable install, already done)

## Skill hints

- NAML course methods → `polimi-naml` skill
- C++ / AMSC patterns → `polimi-amsc` skill
- Statistics notebooks → `polimi-appstat` skill
- PDE / deal.II → `polimi-nmpde` skill
- NLA solvers → `polimi-nla` skill
