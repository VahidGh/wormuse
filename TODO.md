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
| **Status** | 🟡 In Progress |
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
| **Status** | 🔴 Open |
| **Priority** | P0 |
| **Severity** | Correctness — headline result "57.7%" is misleading |

**Description:** `biological_ceiling()` in `midi_target.py` uses a greedy single-voice
algorithm: checks if consecutive Chopin notes are ≥ τ_refrac apart. With 8 or 95 independent
voices, the real capacity is `n_voices × (1/τ_refrac)` >> Chopin's 2.74 notes/s.
The true ceiling is far higher; the real bottleneck is **rhythmic regularity** not note rate.

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
| **Status** | 🔴 Open |
| **Priority** | P2 |
| **Severity** | Biological realism / musical quality |

**Description:** With `n_muscles=95` and `drive_amplitude=8.0`, every muscle crosses
`Ca_THRESH=-10mV` at its wave crest → 38 notes/s (14× Chopin's 2.74/s).
Selective gating (only wave-crest muscles fire) requires either raising `Ca_THRESH` or
lowering `drive_amplitude`.

**Affected files:**
- `PyANNOW/src/pyannow/composer/worm_optimizer_fast.py` — add `Ca_THRESH` parameter, or auto-tune based on `n_muscles`
- `PyANNOW/src/pyannow/composer/worm_optimizer.py` — same
- `PyANNOW/notebooks/02_chopin_worm_optimizer.ipynb` — §3 forward model demo, §5 random baseline
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — data preparation cell
- `TODO.md` — this entry

**Fix plan:**
- Raise `Ca_THRESH` to `+5 mV`; expected ~7 fires/cycle × 0.4 Hz = 2.8 notes/s ✓
- OR: make `Ca_THRESH` a `CelegansChannelParams` field (PINN-tunable)
- Test: run both modes and compare to Chopin note rate

---

## ISSUE-006 — Dorsal/ventral antiphase not exploited (2× note density opportunity)

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P2 |
| **Severity** | Biological realism / musical richness |

**Description:** The worm's dorsal and ventral muscles fire in antiphase during locomotion.
The current model treats all 95 muscles with a single wave, losing this natural doubling.
With dorsal (0-47) offset by π from ventral (48-94), the effective note rate doubles.

**Affected files:**
- `PyANNOW/src/pyannow/composer/worm_optimizer_fast.py` — split `muscle_phases` into dorsal (offset 0) + ventral (offset π)
- `PyANNOW/docs/PyANNOW_NAML_progression.md` — results table (note rate)
- `TODO.md` — this entry

**Fix plan:**
```python
# muscle_phases for 95 cells (48 dorsal + 47 ventral, antiphase)
dorsal  = np.linspace(0.0, 2*np.pi, 48, endpoint=False)
ventral = np.linspace(np.pi, 3*np.pi, 47, endpoint=False)  # offset by π
muscle_phases = np.concatenate([dorsal, ventral])
```

---

## ISSUE-007 — n_muscles default changed but notebooks use hardcoded n_muscles=8

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P2 |
| **Severity** | Consistency |

**Description:** `run_forward_fast` now defaults to `n_muscles=95` but the notebooks
(`02`, `03`) call it with the old 8-cell assumption (e.g. `MUSCLE_PITCHES` size 8,
`V_muscles[:, j]` over range(8)). They will fail or silently use 8-cell data
with a 95-cell function.

**Affected files:**
- `PyANNOW/notebooks/02_chopin_worm_optimizer.ipynb` — all calls to `run_forward_fast`; §3, §4, §5, §6, §9
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — data prep cell, all forward model calls
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — rebuild with explicit `n_muscles=95`
- `docs/_build_chopin_notebook.py` — same
- `TODO.md` — this entry

**Fix plan:**
1. Decide canonical default: keep `n_muscles=95` everywhere, or make each notebook explicit
2. Update all `run_forward_fast(...)` calls to pass `n_muscles=95` explicitly
3. Update `MUSCLE_PITCHES_95` usage in pitch lookups
4. Re-execute both notebooks

---

## ISSUE-008 — PINN selectivity: threshold as PINN-tunable parameter

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P2 |
| **Severity** | Improvement / biological fidelity |

**Description:** `Ca_THRESH` (EGL-19 firing threshold) is hard-coded in the forward model.
It should be a `CelegansChannelParams` field, making it PINN-tunable. The PINN can then
learn the optimal threshold for musical output — directly connecting to the SC-PINN user project.

**Affected files:**
- `PyANNOW/src/pyannow/ion_channels/celegans_hh.py` — add `Ca_thresh: float = -10.0` to `CelegansChannelParams`
- `PyANNOW/src/pyannow/composer/worm_optimizer_fast.py` — read `p.Ca_thresh` instead of hardcoded -10.0
- `PyANNOW/src/pyannow/step8_pinn/locomotion_pinn.py` — add Ca_thresh to the PINN parameter vector
- `PyANNOW/docs/PyANNOW_NAML_progression.md` — update Step 8 description
- `PyANNOW/presentation/index.html` — "Ion channels as tuning knobs" table
- `TODO.md` — this entry

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
| 🔵 **P3** | ISSUE-010 — Real OpenWorm integration | 1 week | Full biological fidelity |
