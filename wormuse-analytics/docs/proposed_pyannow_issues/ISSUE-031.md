## ISSUE-031 — 8-muscle pitch bottleneck blocks Chopin `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P1 |
| **Severity** | Correctness — caps achievable pitch-aware F1 well below 1.0 |

**Description.** `pyannow/composer/worm_optimizer.py` provides two pitch maps:

```python
MUSCLE_PITCHES    = generate_muscle_pitches(n_muscles=8)   # 8-cell simplified
MUSCLE_PITCHES_95 = generate_muscle_pitches(n_muscles=95)  # 95-cell full BWM
```

Notebook 03 uses the 8-muscle path (`n_muscles=8` is the default of `run_forward_fast`). Chopin's Nocturne in C♯ minor uses ~80 distinct pitches in the first 10 seconds alone. The 8 pentatonic pitches cover at most ~50 % of the Chopin pitch-class set; any worm with `n_muscles=8` therefore has a hard upper bound on **pitch-aware F1** well below 1.0.

This is invisible while the metric is `onset_loss` or plain `musical_f1` (both pitch-blind). Once `pitch_aware_f1` (ISSUE-035) is reported, the ceiling becomes the headline question.

**Compute the ceiling.** `wormuse_analytics.pipeline.reachable_pitches(muscle_pitches, chopin_pitches)` returns the fraction of Chopin pitches reachable by pitch-class match and the corresponding F1 ceiling.

**Fix plan.**

1. **Switch the forward model to `n_muscles=95`.** Already supported by `run_forward_fast(p, n_muscles=95)`. No code change; just a parameter:

   ```python
   result_worm = run_forward_fast(DEFAULT_PARAMS, duration_s=DURATION, dt_ms=0.5,
                                   drive_freq_hz=1.5, drive_amplitude=12.0,
                                   n_muscles=95)   # ← was default n_muscles=8
   ```

2. With 95 muscle pitches the reachable-pitch fraction approaches 1.0 (covers all chromatic pitch-classes); the F1 ceiling becomes near-perfect.

3. **Report the ceiling in cell 24 summary table.** Add a row "Pitch ceiling (current map)" with the reachable_pitch_aware_f1 value.

**Caveats.**

- Step 8 PINN expects per-muscle voltage matrices; verify it works at `n_muscles=95`.
- Step 5's `selective gating` for the 95-cell model is itself ISSUE-005 (already open in PyANNOW) — until that lands, the 95-muscle path may overproduce notes. Document this dependency.

**Affected files.**
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — cell 4 set `n_muscles=95`.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/src/pyannow/targets/midi_target.py` — add `reachable_pitch_set` helper (reference impl in wormuse_analytics.pipeline.reachable_pitches).
- `PyANNOW/TODO.md` — this entry. Also reference ISSUE-005 (selective gating dependency).
