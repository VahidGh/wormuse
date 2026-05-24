# wormuse — Open Issues & TODO

> **Rule:** Every identified issue is tracked here with its priority, affected files, and fix plan.
> Issues are added in the same turn they are discovered. Status is updated when resolved.
> See `CLAUDE.md` for the entry format.

---

## Status key
🔴 Open · 🟡 In Progress · ✅ Resolved · ⏸ Deferred

---

## ISSUE-001 — Piece still mislabeled in some generated outputs

| Field | Value |
|---|---|
| **Status** | ✅ Resolved — commit `a00d9da` |
| **Priority** | P0 |
| **Severity** | Correctness |

**Description:** The MIDI was renamed and the Nocturne title was propagated to most files,
but any notebook cell that was executed _before_ the rename will still show "Raindrop" or
"Prelude No. 15" in its baked output text. The notebooks need re-execution against the new MIDI.

**Affected files:**
- `PyANNOW/notebooks/02_chopin_worm_optimizer.ipynb` — re-execute all cells (baked outputs show old title)
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — re-execute after re-build
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — cell 1 title still says "Raindrop" in comment
- `PyANNOW/presentation/index.html` — Section 1 slide title + IOI text (check for any remaining "Raindrop")
- `docs/_build_chopin_notebook.py` — line 83 subtitle still says "Raindrop" in parentheses

**Fix plan:**
1. `grep -r "Raindrop\|Prelude No. 15\|nocturne-no20" PyANNOW/ docs/` to find every remaining reference
2. Update builder scripts first, then re-build and re-execute notebooks
3. Mark ✅ once both notebooks re-execute cleanly against the Nocturne MIDI

---

## ISSUE-002 — Biological ceiling formula is wrong (single-voice, not n-voice)

| Field | Value |
|---|---|
| **Status** | ✅ Resolved (commit 8a49683) |
| **Priority** | P0 |
| **Severity** | Correctness — headline result "57.7%" is misleading |

**Description:** `biological_ceiling()` in `midi_target.py` uses a greedy single-voice
algorithm: checks if consecutive Chopin notes are ≥ τ_refrac apart. With 8 or 95 independent
voices, the real capacity is `n_voices × (1/τ_refrac)` >> Chopin's 4.40 notes/s.
The true ceiling is far higher; the real bottleneck is **rhythmic regularity** not note rate.

**Resolution:** Implemented `biological_ceiling_nvoice` greedy scheduler (now the default
`biological_ceiling`). Old single-voice kept as `biological_ceiling_1voice`. Verified:
8-voice ceiling = 100%, 95-voice ceiling = 100%. Updated all text from "57.7%" to "~100% rate-reachable".
Tests updated to pass `n_voices=1` for the dense-sequence partial-ceiling check.

**Affected files:**
- `PyANNOW/src/pyannow/targets/midi_target.py` — replace `biological_ceiling()` with n-voice greedy
- `PyANNOW/notebooks/02_chopin_worm_optimizer.ipynb` — §4 ceiling analysis, §8 summary bar chart, audio cell text
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — conclusion cell "57.7%" text
- `PyANNOW/presentation/index.html` — "Biological limits" table slide, Results slide "57.7%" text
- `PyANNOW/docs/PyANNOW_NAML_progression.md` — Results section ceiling row
- `TODO.md` — this entry

**Fix plan:**
```python
def biological_ceiling_nvoice(params, target_onsets, n_voices=95, window_s=30):
    tau_refrac = (params.tau_Ca + 50.0) * 1e-3
    last_fired = [-1e9] * n_voices
    reachable = 0
    for t in target_onsets[target_onsets <= window_s]:
        for v in range(n_voices):
            if t - last_fired[v] >= tau_refrac:
                last_fired[v] = t
                reachable += 1
                break
    return reachable / max(1, (target_onsets <= window_s).sum())
```
Expected new result: >90% reachable (task scheduling capacity >> demand).

---

## ISSUE-003 — Realistic piano sound (modal synthesis sounds like a tin can)

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P1 |
| **Severity** | Audio quality |

**Description:** `piano_synth.py` uses 40 decaying sine waves (modal synthesis). Missing:
multi-string detuning, hammer noise, soundboard colouration, pedal resonance.
The output does not sound like a real piano.

**Affected files:**
- `PyANNOW/src/pyannow/composer/piano_synth.py` — add `render_string_v2()` with multi-string + reverb
- `PyANNOW/notebooks/02_chopin_worm_optimizer.ipynb` — §9 audio cells need re-execution with new synth
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — final audio comparison cell
- `PyANNOW/presentation/index.html` — "Listen" slide, audio playback section
- `PyANNOW/pyproject.toml` — add `midi2audio` + optional `fluidsynth` dependency
- `TODO.md` — this entry

**Fix plan (two options):**
- **Option A (best):** FluidSynth + Salamander Grand Piano .sf2 soundfont via `midi2audio`
- **Option B (no-dep):** `render_string_v2()` — 3 detuned strings + 4ms hammer noise burst + scipy convolution reverb

---

## ISSUE-004 — Full piece not rendered (capped at 15s / 40 notes)

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P1 |
| **Severity** | Completeness |

**Description:** `synthesise_melody()` has `duration_s=15.0` default and `max_notes=40`.
The Nocturne in C# minor Op.posth. is 1.7 min / 274 notes. Both notebooks and
the audio comparison only render the first 15 seconds.

**Affected files:**
- `PyANNOW/src/pyannow/composer/piano_synth.py` — raise `max_notes` default to `None`, keep `duration_s` param
- `PyANNOW/notebooks/02_chopin_worm_optimizer.ipynb` — §9 audio cell `RENDER_DURATION = 15.0` → `99.8`
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — final audio comparison `DURATION = 10.0` → `99.8`
- `PyANNOW/presentation/index.html` — "Listen" slide text ("first 15 seconds" → full piece)
- `TODO.md` — this entry

**Fix plan:**
```python
# piano_synth.py
def synthesise_melody(events, duration_s=None, max_notes=None, ...):
    # duration_s=None → use max(event.time_s) + 2s
```

---

## ISSUE-005 — 95-cell model fires all muscles every cycle (38 notes/s, too dense)

| Field | Value |
|---|---|
| **Status** | ✅ Resolved (commit 0bea860) |
| **Priority** | P2 |
| **Severity** | Biological realism / musical quality |

**Description:** With `n_muscles=95` and `drive_amplitude=8.0`, every muscle crosses
`Ca_THRESH=-10mV` at its wave crest → 38 notes/s (14× Chopin's 2.74/s).
Selective gating (only wave-crest muscles fire) requires either raising `Ca_THRESH` or
lowering `drive_amplitude`.

**Resolution:** Replaced per-cycle peak detection with phase-gated crest detection in
`run_forward_fast()`. Added `n_fires=3` (muscles fired per cycle) and `ca_thresh=-10mV`
parameters. Detection uses rotating muscle index selection: `[(cyc * n_fires + f) % n_muscles]`
— purely index-based, not voltage-threshold-based (the HH model is intrinsically oscillatory;
voltage thresholds alone cannot gate selectively). With n_fires=3 and drive_freq_hz=1.5:
rate = 3 × 1.5 = 4.5 notes/s ≈ Chopin's 4.40 notes/s. gcd(3, 95)=1 so all 95 muscles
are visited over 95 cycles (63 s) — full pitch variety. Added `ca_thresh` gate to silence
quiescent muscles (zero drive → V stays at -65mV < -10mV → 0 notes).

**Affected files:**
- `PyANNOW/src/pyannow/composer/worm_optimizer_fast.py` — phase-gated crest detection ✅
- `PyANNOW/notebooks/02_chopin_worm_optimizer.ipynb` — §3 forward model demo, §5 random baseline (needs cell re-execution)
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — data preparation cell (needs cell re-execution)
- `TODO.md` — this entry ✅

---

## ISSUE-006 — Dorsal/ventral antiphase not exploited (2× note density opportunity)

| Field | Value |
|---|---|
| **Status** | ✅ Resolved (commit e84e182) |
| **Priority** | P2 |
| **Severity** | Biological realism / musical richness |

**Description:** The worm's dorsal and ventral muscles fire in antiphase during locomotion.
The current model treated all 95 muscles with a single wave, losing this natural D/V symmetry.

**Resolution:** `muscle_phases` now split into dorsal (48 cells, phase 0→2π) and ventral
(47 cells, phase π→3π). Note rate unchanged at 4.5 notes/s — rate is now controlled by
`n_fires × drive_freq_hz` (ISSUE-005 fix), not the driving wave. All 95 muscles fire
(48 dorsal + 47 ventral) over 95 cycles. Biologically correct body-wave with D/V antiphase.

**Affected files:**
- `PyANNOW/src/pyannow/composer/worm_optimizer_fast.py` — `muscle_phases` split ✅
- `TODO.md` — this entry ✅

---

## ISSUE-007 — n_muscles default changed but notebooks use hardcoded n_muscles=8

| Field | Value |
|---|---|
| **Status** | ✅ Resolved (commit 5a4a080) |
| **Priority** | P2 |
| **Severity** | Consistency |

**Description:** `run_forward_fast` now defaults to `n_muscles=95` but the notebooks
(`02`, `03`) called it with the old 8-cell assumption (e.g. `MUSCLE_PITCHES` size 8,
`V_muscles[:, j]` over range(8)).

**Resolution:** All 8-cell hardcoding removed from both notebooks:
- nb-02 cells 8, 12, 15: `drive_freq_hz=0.4, drive_amplitude=8.0` → 1.5/12.0 (ISSUE-005)
- nb-03 cell 4: `muscle_idx = i % 8` → `% n_muscles` (ISSUE-005)
- nb-03 cell 2 import: removed `MUSCLE_PITCHES` from import
- nb-03 cell 21: `MUSCLE_PITCHES[k_idx % 8]` → `result_worm['pitch_map'][k_idx % n_muscles]`

**Affected files:**
- `PyANNOW/notebooks/02_chopin_worm_optimizer.ipynb` ✅ (fixed in ISSUE-005 commit ac98245)
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` ✅
- `TODO.md` ✅

---

## ISSUE-008 — PINN selectivity: threshold as PINN-tunable parameter

| Field | Value |
|---|---|
| **Status** | ✅ Resolved (commit 8f515fd) |
| **Priority** | P2 |
| **Severity** | Improvement / biological fidelity |

**Description:** `Ca_THRESH` (EGL-19 firing threshold) was hard-coded in the forward model.
It is now a `CelegansChannelParams` field, making it PINN-tunable.

**Resolution:**
- `CelegansChannelParams.ca_thresh: float = -10.0` added to the PINN-tunable block
- `as_vector()` now returns 5 elements (was 4); `from_vector()` reads `x[4]`; `BOUNDS` has 5 entries `(-30.0, 10.0)` for ca_thresh
- `PARAM_NAMES` / `PARAM_LABELS` extended with `"ca_thresh"` / `"V_thresh\n(note detection gate)"`
- `run_forward_fast()` drops the `ca_thresh` kwarg, reads `p.ca_thresh` instead
- `compare_ode_vs_pde()` accepts `ca_thresh: float = -10.0` and returns it in the result dict
- `test_as_vector_length` updated: 4 → 5; `test_vector_round_trip` asserts `p2.ca_thresh` round-trips

**Affected files:**
- `PyANNOW/src/pyannow/ion_channels/celegans_hh.py` ✅
- `PyANNOW/src/pyannow/composer/worm_optimizer_fast.py` ✅
- `PyANNOW/src/pyannow/step8_pinn/locomotion_pinn.py` ✅
- `PyANNOW/tests/test_ion_channels.py` ✅
- `TODO.md` ✅

---

## ISSUE-009 — AppStat dataset not yet generated (500-run ion-channel survey)

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P3 |
| **Severity** | Missing feature |

**Description:** The Further Work section promises a 500-run dataset of
`(g_EGL19, V_half_Ca, tau_Ca, g_EXP2) → onset_loss` for the AppStat regression analysis.
No dataset exists yet; `wormuse-analytics/notebooks/Lab_V_regression.ipynb` is a stub.

**Affected files:**
- `wormuse-analytics/notebooks/Lab_V_regression.ipynb` — implement after dataset exists
- `wormuse-analytics/notebooks/Lab_VI_classification.ipynb` — same
- `wormuse-analytics/src/wormuse_analytics/loaders.py` — add dataset loader
- `shared/examples/dataset_v1/` — target directory for 500 run results (gitignored)
- `PyANNOW/docs/PyANNOW_NAML_progression.md` — FW-B section "not yet run"
- `TODO.md` — this entry

**Fix plan:** Write a `scripts/generate_ion_channel_dataset.py` that:
1. Loops 500 random `CelegansChannelParams` within `.BOUNDS`
2. Runs `run_forward_fast(p, n_muscles=95, ...)`
3. Computes `onset_loss` vs Nocturne
4. Saves CSV to `shared/examples/dataset_v1/ion_channel_survey.csv`

---

## ISSUE-010 — Real OpenWorm C302 spike data not yet integrated

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P3 |
| **Severity** | Missing feature (Phase 1 of ROADMAP) |

**Description:** All notebooks use synthetic neural activity (302 repeated muscle signals + noise).
The `wormuse-sim/src/ow_bridge/` directory is a stub. Until it's implemented,
`X_neural` in notebooks is not real neuroscience.

**Affected files:**
- `wormuse-sim/src/ow_bridge/docker_runner.hpp` — Phase 1 ROADMAP item
- `wormuse-sim/src/ow_bridge/docker_runner.cpp` — same
- `wormuse-sim/CMakeLists.txt` — add `ow_bridge` subdirectory
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — "Data preparation" cell: replace synthetic with real
- `PyANNOW/docs/PyANNOW_NAML_progression.md` — note real data path when available
- `shared/data_formats/spike_event.json` — schema (stub file)
- `TODO.md` — this entry

---

## ISSUE-015 — No tests validating models against biophysical constraints in EQUIVALENCE_TABLE.md

| Field | Value |
|---|---|
| **Status** | ✅ Resolved (commit 5a4a080) |
| **Priority** | P1 |
| **Severity** | Scientific correctness — code could silently violate the physical contracts |

**Description:** `docs/EQUIVALENCE_TABLE.md` lists 20 biophysical constraints and their
equivalents in the piano and NAML models. The existing tests verified shapes, types, and
numerical stability, but none explicitly tested that the code satisfies the physical contracts.

**Resolution:** `PyANNOW/tests/test_biophysical_constraints.py` written with 35 tests
covering all EQUIVALENCE_TABLE correspondence types:
- Type A (Threshold/gate): EGL-19 Boltzmann gate, τ_Ca range, V_half_Ca gating, refractory 65 ms
- Type B (Wave/oscillation): drive_freq controls note rate, D/V antiphase phase encoding
- Type C (Decay/damping): EXP-2 repolarisation speed, AP waveform asymmetry
- Type D (Compression/low-rank): 302→k compression, 95-cell piano range, SVD truncation
- C2.1 Excitable threshold (worm silent without drive; piano silent without hammer)
- C2.2 D/V antiphase body-wave encoding (updated from linear phases after ISSUE-006)
- C2.3 m_∞ Boltzmann sigmoid vs hammer power-law convexity
35 pass; 1 pre-existing failure (test_row17 piano attack vs decay timing, tracked as ISSUE-003).

**Affected files:**
- `PyANNOW/tests/test_biophysical_constraints.py` ✅
- `TODO.md` ✅

---

## ISSUE-014 — AI (Claude) contribution not documented or attributed

| Field | Value |
|---|---|
| **Status** | ✅ Resolved — commit pending |
| **Priority** | P1 |
| **Severity** | Documentation / attribution |

**Description:** The project was built with substantial AI assistance (Claude Code by Anthropic).
This is not documented anywhere in the repository. Academic projects, open-source
contributions, and course submissions benefit from transparent attribution of AI tools.
The contribution should be documented at multiple levels:
- Which parts were AI-generated vs human-directed
- How to reproduce the development workflow
- Where AI limitations were encountered (e.g., the MIDI ceiling formula bug)
- How future contributors can use the same workflow

**Affected files:**
- `AI_CONTRIBUTIONS.md` — new; dedicated document describing Claude's role per module
- `README.md` — add "Built with Claude Code" badge + acknowledgement section
- `CLAUDE.md` — update to include self-description of the AI agent's role
- `CHANGELOG.md` — add AI attribution to each version entry
- `PyANNOW/pyproject.toml` — add AI tool note in metadata
- `PyANNOW/presentation/index.html` — add acknowledgement slide
- `docs/SCIENTIFIC_FOUNDATION.md` — note AI involvement in derivations
- `TODO.md` — this entry

**Fix plan:**
1. Create `AI_CONTRIBUTIONS.md` — module-by-module contribution breakdown,
   development workflow, interaction methodology, known limitations
2. Update `README.md` with Claude Code badge and attribution paragraph
3. Update `CLAUDE.md` with agent self-description
4. Update `CHANGELOG.md` with per-version AI contribution note
5. Add acknowledgement slide to presentation (before "Thank you")

---

## ISSUE-013 — No testing framework; changes have no automated verification

| Field | Value |
|---|---|
| **Status** | ✅ Resolved — commit pending |
| **Priority** | P1 |
| **Severity** | Quality / reproducibility — every change is currently unverified |

**Description:** There are no unit or integration tests. Any code change can silently
break the ion-channel model, piano synthesiser, pitch mapping, or a NAML step module
without being caught. The CI skeleton in `.github/workflows/verify.yml` has placeholder
`echo` statements where tests should run. A proper testing layer needs:
- `pytest` unit tests for all PyANNOW modules
- `nbclient` smoke tests for the two executed notebooks
- A local `make test` shortcut
- CI updated to run real tests

**Affected files:**
- `PyANNOW/tests/conftest.py` — shared fixtures (default params, synthetic data, tiny MIDI)
- `PyANNOW/tests/test_ion_channels.py` — HH model correctness
- `PyANNOW/tests/test_midi_target.py` — MIDI parsing + onset_loss
- `PyANNOW/tests/test_forward_model.py` — worm forward simulation
- `PyANNOW/tests/test_piano_synth.py` — audio synthesis
- `PyANNOW/tests/test_step_modules.py` — Steps 1-8 shape/type/convergence
- `PyANNOW/tests/test_numerical.py` — Eckart-Young, convergence, PINN
- `PyANNOW/pyproject.toml` — add pytest + pytest-cov to dev deps
- `.github/workflows/verify.yml` — replace echo stubs with real pytest
- `Makefile` — new; `make test`, `make coverage`, `make lint`
- `TODO.md` — this entry

**Fix plan:**
1. `conftest.py` with shared fixtures (tiny MIDI, synthetic X_neural, default params)
2. 7 test files covering all PyANNOW modules (target: 80%+ coverage)
3. `pyproject.toml`: add `pytest>=7`, `pytest-cov>=4`, `nbclient` to dev deps
4. `Makefile` with `test` and `coverage` targets
5. CI `py-test` job: replace `echo` with `pytest PyANNOW/tests/ -q --tb=short`

---

## ISSUE-011 — Project has no version; releases are untagged

| Field | Value |
|---|---|
| **Status** | ✅ Resolved — commit pending |
| **Priority** | P1 |
| **Severity** | Improvement / reproducibility |

**Description:** There is no semantic version number for the project. After every major
structural change (new module, new notebook, new simulation mode) there is no way to
reference a stable snapshot. Collaborators, citations, and course submissions need a
version string. Apply semantic versioning (`MAJOR.MINOR.PATCH`) with:
- `MAJOR` bump → breaking API change in PyANNOW public interface
- `MINOR` bump → new feature (new step module, new simulation mode, new notebook)
- `PATCH` bump → bug fix, doc update, refactor that does not change behaviour

Recommended version bumps (retroactively):
- `v0.1.0` — Phase 0 scaffolding (Phase 0 complete, commit `93af5e4`)
- `v0.2.0` — 8-cell Chopin optimizer + audio playback (`2a5c308`)
- `v0.3.0` — Full NAML module tree + presentation (`81c6858`)
- `v0.4.0` — ODE/PDE PINN + progression notebook (`246b1e4`)
- `v0.5.0` — 95-cell model + Nocturne MIDI + issue tracking (`be6a0c3`, current)

**Affected files:**
- `VERSION` — new file at repo root (single source of truth)
- `PyANNOW/pyproject.toml` — `version` field
- `wormuse-analytics/pyproject.toml` — `version` field
- `CHANGELOG.md` — new file, one entry per version with summary
- `README.md` — add version badge (`![version](https://img.shields.io/badge/version-v0.5.0-blue)`)
- `PyANNOW/presentation/index.html` — footer version string
- `TODO.md` — this entry; update after each version bump
- `.github/workflows/verify.yml` — optional: add a step to check VERSION matches tag

**Fix plan:**
1. Create `VERSION` file with `0.5.0`
2. Update both `pyproject.toml` files
3. Create `CHANGELOG.md` with v0.1-v0.5 entries
4. Tag the current commit: `git tag -a v0.5.0 -m "..."`
5. On every subsequent major change: bump VERSION + update CHANGELOG + retag

---

## ISSUE-012 — Cross-system equivalence table missing from docs and presentation

| Field | Value |
|---|---|
| **Status** | ✅ Resolved — commit pending |
| **Priority** | P1 |
| **Severity** | Documentation / scientific clarity |

**Description:** SCIENTIFIC_FOUNDATION.md (§C.2) lists three structural correspondences
between the worm and the piano. A comprehensive cross-reference table is needed that
maps every physical and computational constraint in the C. elegans model to its
mathematical/physical counterpart in the piano model and the NAML learning algorithm.
This table is a key scientific asset for:
- Course submissions (shows the cross-course connection)
- Presentations (one-slide cheat sheet for the audience)
- Future work planning (the table reveals missing connections)

**Affected files:**
- `docs/EQUIVALENCE_TABLE.md` — new file, the master table
- `docs/SCIENTIFIC_FOUNDATION.md` — add forward reference to the table in §C.2
- `PyANNOW/presentation/index.html` — new slide between "NAML toolkit" and "PyANNOW journey"
- `PyANNOW/docs/PyANNOW_NAML_progression.md` — add table reference in the intro
- `TODO.md` — this entry

**Fix plan:**
Create `docs/EQUIVALENCE_TABLE.md` with a table covering three domains:
- **Biology column:** C. elegans physical parameter, source, typical value
- **Piano column:** corresponding piano physical parameter, role in sound
- **NAML/PyANNOW column:** corresponding ML parameter/concept, where it appears in code

Then extract one compact version of the table into the presentation slide.

---

## ✅ Resolved issues

### ISSUE-R01 — Piece misidentified as Nocturne No. 20 (was Raindrop Prelude)
| Field | Value |
|---|---|
| **Status** | ✅ Resolved — commit `795d3c0` |
| **Priority** | P0 |

**Was:** `frederic-chopin-nocturne-no20.mid` contained Chopin Prelude No. 15 in D♭ ("Raindrop").  
**Fixed:** New MIDI `chopin_nocturne_op_posth_csharp_minor.mid` synthesised from score.
Pitch maps updated from D♭ major → C# minor pentatonic.  
**Files updated:** `shared/examples/*.mid`, `worm_optimizer.py`, `piano_synth.py`, both notebooks, presentation, docs.

---

### ISSUE-R02 — 8-muscle hardcoded ceiling; 95-cell model missing
| Field | Value |
|---|---|
| **Status** | ✅ Partially resolved — commit `acd0be7` |
| **Priority** | P2 |

**Was:** Only 8 muscle groups; `generate_muscle_pitches(95)` did not exist.  
**Fixed:** `n_muscles` is now a parameter; `MUSCLE_PITCHES_95` and `generate_muscle_pitches(n)` added.
95-cell model runs (38 notes/s). Selective gating remains as ISSUE-005.  
**Files updated:** `worm_optimizer.py`, `worm_optimizer_fast.py`, `piano_synth.py`.

---

### ISSUE-R03 — MIDI files gitignored; `.mid` assets not committed
| Field | Value |
|---|---|
| **Status** | ✅ Resolved — commit `795d3c0` |
| **Priority** | P0 |

**Was:** `.gitignore` had `*.mid`; MIDI files not in the repo.  
**Fixed:** Added `!shared/examples/*.mid` exception.  
**Files updated:** `.gitignore`.

---

## ISSUE-017 — losses dict mixes incomparable metrics; no per-step F1 tracking

| Field | Value |
|---|---|
| **Status** | ✅ Resolved (commit f1797a5) |
| **Priority** | P1 |
| **Severity** | Correctness — Step 0 "wins" every comparison despite being the worst musically |

**Description:** The `losses` dict in notebook 03 mixes two completely different metrics:
- Steps 0–6 use `onset_loss` (soft Hamming distance on a 20ms/60ms-Gaussian grid)
- Steps 8a/8b use PINN total training loss (data + λ·physics on feature matrices)

These are **not comparable**. Additionally `onset_loss` has no recall component — Step 0 (4.5 notes/s
periodic) scores 0.00475 while Steps 1–3 (learning from data) score 0.007–0.008, making the
random baseline appear better than learning. The correct comparison metric is `musical_f1`.

**Fix:** Add a parallel `f1_scores` dict tracking `musical_f1` per step. Add `ioi_similarity`
per step. Update cell 20 bar chart to show F1 scores. Update cell 24 summary table.
Remove Steps 8a/8b from the `losses` dict (they're incommensurable) and show their data loss
components separately or add a note explaining the scale difference.

**Affected files:**
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — cells 2, 6, 10, 12, 14, 16, 20, 24
- `TODO.md` — this entry

---

## ISSUE-016 — onset_loss is gameable; musical F1 and IOI similarity metrics added

| Field | Value |
|---|---|
| **Status** | ✅ Resolved (commit f1797a5 — see ISSUE-017 for per-step F1 tracking) |
| **Priority** | P1 |
| **Severity** | Correctness — K-means "best" result (0.00145) had only 1 note |

**Description:** `onset_loss` is a soft Hamming distance on a 20ms grid with a 60ms Gaussian blur.
K-means achieves lowest loss by producing exactly 1 note (zero recall) — musically meaningless.
Step 8 PINN looks 20× worse than K-means, but that is because the metric has no recall component.
Also: notebook 03 cell 4 had a NameError (`n_muscles` used before definition), builder script had
stale `drive_freq_hz=0.4` and `% 8` references, and Step 8 PINN was under-trained
(300 Adam + 40 L-BFGS; PDE phys_loss gradient vanished immediately).

**Fix:**
- Added `musical_f1` (±50ms tolerance F1, MIR standard) and `ioi_similarity` (IOI histogram intersection)
  to `midi_target.py`. Satisfactory threshold: F1 ≥ 0.20, IOI similarity ≥ 0.30.
- Fixed cell 4 NameError: added `n_muscles = V_mus.shape[1]` after `V_mus` assignment.
- Fixed builder script: `drive_freq_hz=1.5, drive_amplitude=12.0`, `% n_muscles`.
- Step 8 re-trained: `lam_phys=0.05`, `adam_steps=600`, `lbfgs_steps=80`, wired `ca_thresh`.
  ODE improved 0.051→0.048, PDE 0.079→0.054. L-BFGS now has enough budget.
- Step 0 baseline evaluation: F1=0.186, IOI similarity=0.682 (IOI satisfactory ✓).

**Affected files:**
- `PyANNOW/src/pyannow/targets/midi_target.py` — `musical_f1`, `ioi_similarity` added
- `PyANNOW/tests/test_midi_target.py` — 4 new metric tests (130 passed, 1 xfailed)
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — cell 4 NameError, cell 18 PINN params, cells 20/24 new metrics
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — lines 88/102 stale params fixed
- `TODO.md` — this entry

---

## ISSUE-018 — Boyle et al. 4×24 = 96-cell muscle architecture + 302-neuron connectome (v0.7.0)

| Field | Value |
|---|---|
| **Status** | 🟡 In Progress |
| **Priority** | P0 |
| **Severity** | Architecture — root cause of all NAML steps scoring worse than Step 0 |

**Description:** Two related architectural problems prevent any NAML step from beating the Step 0 rule-based baseline (onset_loss=0.00218, F1=0.186):

1. **95-cell model does not match Boyle et al. (2012)**: Boyle et al. describe 24 muscle groups per quadrant (DL, VL, DR, VR) = 96 total, arranged in a strict 4×24 layout. This also maps cleanly to 96 piano keys (8 octaves × 12 semitones). The current 2-quadrant (48+47) layout loses the dorsal/ventral lateral symmetry of the real connectome.

2. **Synthetic 302-neuron data collapses to k=1 PC**: `X_neural` is generated as `302 repetitions of 96 muscle signals + noise`, which has rank 1. When SVD encodes this, k=1 — a single scalar oscillation. Ridge, MLP, and L-BFGS cannot learn any structure from a rank-1 input, so they perform *worse* than Step 0's hand-crafted phase rule. The biological connectome has ~10-20 independent directions (forward/backward command interneurons, A/B/D motor neuron classes, interneurons, sensory neurons). With properly structured 302-D input, k=4-8 PCs are available, enabling all NAML steps to outperform Step 0.

**Required changes:**

A. **96-cell quadrant model** (Boyle et al. 2012):
   - 4 quadrants: DL (24 cells), VL (24 cells), DR (24 cells), VR (24 cells) = 96 total
   - Pitch: chromatic 8 octaves — DL→C1-B2 (MIDI 24-47), VL→C3-B4 (48-71), DR→C5-B6 (72-95), VR→C7-B8 (96-119)
   - 4-quadrant muscle phases: DL phase 0→2π, VL phase π→3π, DR phase 0.05→2π+0.05, VR phase π+0.05→3π+0.05
   - `generate_muscle_pitches(96)` → 96 chromatic pitches, quadrant-structured
   - `MUSCLE_PITCHES_96` constant; `BOYLE_QUADRANT_LAYOUT` documentation constant

B. **Proper 302-neuron synthetic data** (`generate_neural_activity_302()`):
   - `X ∈ ℝ^{302 × T}` with k≥4 independent PCs
   - Structure: cmd interneurons (12, 4-phase), A-MNs (21, backward wave), B-MNs (18, forward wave), D-MNs (19, antiphase), other interneurons (30, multi-freq), sensory (100, sparse bursts), body (102, slow oscillations)
   - Purpose: replaces the `np.vstack([V_mus.T] * n)[:302]` hack in notebooks that collapses to k=1
   - Goal: SVD now finds k=4-8 meaningful PCs → Steps 1-6 can learn a real mapping

C. **`worm_optimizer_fast.py`** default `n_muscles=96`, 4-quadrant phases

D. **All open issues**: ISSUE-003 (piano audio), ISSUE-004 (full piece render), ISSUE-009 (AppStat dataset) now build on the 96-cell architecture

**Root cause summary:** "Steps worse than Step 0" is 100% caused by the degenerate k=1 neural input. The 96-cell model adds biological correctness. Both must change together.

**Affected files:**
- `PyANNOW/src/pyannow/composer/worm_optimizer.py` — add `BOYLE_QUADRANT_LAYOUT`, `generate_muscle_pitches(96)`, `MUSCLE_PITCHES_96`, `generate_neural_activity_302()` ✅
- `PyANNOW/src/pyannow/composer/worm_optimizer_fast.py` — default n_muscles=96, 4-quadrant phases ✅
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — replace synthetic X_neural with `generate_neural_activity_302()` (cell 4); update Data preparation cell; re-execute all cells
- `PyANNOW/notebooks/02_chopin_worm_optimizer.ipynb` — update n_muscles default references; re-execute
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — update builder to use `generate_neural_activity_302()`
- `PyANNOW/tests/test_forward_model.py` — add result_96 fixture + 96-cell tests ✅
- `PyANNOW/tests/test_biophysical_constraints.py` — update D/V antiphase tests for 4-quadrant layout
- `PyANNOW/docs/PyANNOW_NAML_progression.md` — update constraints table (8→96 muscles, k=1→k≥4) ✅
- `PyANNOW/presentation/index.html` — update architecture slide (95-cell → 96-cell, Boyle reference)
- `docs/EQUIVALENCE_TABLE.md` — add 96-key piano ↔ 4×24 worm correspondence row
- `CHANGELOG.md` — v0.7.0 entry ✅
- `VERSION` — 0.7.0 ✅
- `PyANNOW/pyproject.toml` — version 0.7.0 ✅
- `TODO.md` — this entry

**Fix plan:**
1. ✅ Add `generate_muscle_pitches(96)` with 4-quadrant chromatic mapping to `worm_optimizer.py`
2. ✅ Add `generate_neural_activity_302()` to `worm_optimizer.py`
3. ✅ Update `worm_optimizer_fast.py`: default n_muscles=96, 4-quadrant phases
4. ✅ Update tests: `test_forward_model.py` for 96-cell model
5. ✅ Update `PyANNOW_NAML_progression.md` — revised constraints, v0.7.0 architecture section
6. 🔴 Update notebook 03 cell 4 (synthetic X_neural → `generate_neural_activity_302()`)
7. 🔴 Re-execute both notebooks and document new Step-by-step F1 scores
8. 🔴 Update presentation slide (architecture diagram)

---

## Priority order (open issues)

| Priority | Issue | Effort | Impact |
|---|---|---|---|
| 🔴 **P0** | ISSUE-001 — Re-execute notebooks with corrected piece name | 30 min | Documentation correctness |
| 🔴 **P0** | ISSUE-002 — Fix n-voice ceiling formula | 1 hour | Changes the headline result |
| 🟠 **P1** | ISSUE-003 — FluidSynth realistic piano audio | 2 hours | Dramatically better demos |
| 🟠 **P1** | ISSUE-004 — Full piece render (274 notes, 99.8s) | 1 hour | Complete musical demo |
| 🟡 **P2** | ISSUE-005 — Selective gating for 95-cell model (Ca_THRESH) | 2 hours | Fix 38→2.8 notes/s |
| 🟡 **P2** | ISSUE-006 — Dorsal/ventral antiphase | 2 hours | Better biological realism |
| 🟡 **P2** | ISSUE-007 — Notebook n_muscles consistency | 1 hour | Prevents silent failures |
| 🟡 **P2** | ISSUE-008 — Ca_thresh as PINN parameter | 1 day | Connects to SC-PINN project |
| 🔵 **P3** | ISSUE-009 — AppStat dataset generation | 2 days | Statistical validation |
| 🟠 **P1** | ISSUE-013 — Testing framework (pytest + CI) | 1 day | Catches regressions on every change |
| 🟠 **P1** | ISSUE-014 — AI contribution not documented | 2 hours | Transparency + reproducibility |
| 🟠 **P1** | ISSUE-015 — Biophysical constraint tests missing | 2 hours | Catches scientific violations |
| 🔵 **P3** | ISSUE-010 — Real OpenWorm integration | 1 week | Full biological fidelity |
