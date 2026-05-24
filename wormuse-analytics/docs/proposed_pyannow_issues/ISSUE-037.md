## ISSUE-037 — "Biological ceiling" formula assumes any muscle can fire any pitch `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P2 |
| **Severity** | Correctness — ceiling overstated; misleads expectations |

**Description.** `pyannow/targets/midi_target.py:biological_ceiling`:

```python
def biological_ceiling(p_celegans, target_onsets, window_s=30.0, n_voices=95):
    tau_refrac = (p_celegans.tau_Ca + 50.0) * 1e-3
    last_fired = np.full(n_voices, -np.inf)
    reachable = 0
    for t in t_clip:
        avail = np.where(last_fired + tau_refrac <= t)[0]
        if len(avail) > 0:
            best = avail[np.argmax(last_fired[avail])]
            last_fired[best] = t
            reachable += 1
    return {"reachable_fraction": reachable / len(t_clip), ...}
```

This greedy scheduler assigns each Chopin note to a voice with enough refractory time. It assumes **any of the 95 voices can produce any of Chopin's pitches** — which is false in the implemented system. Each muscle has a **fixed** pitch via `MUSCLE_PITCHES`; muscle k *always* fires pitch p_k, never pitch p_j.

The ceiling reported is therefore a **rate ceiling** ("can the worm produce notes fast enough?") not a **pitch ceiling** ("can the worm produce the right notes?"). The rate ceiling is ~100% for almost any reasonable τ_Ca; the pitch ceiling, computed against the actual `MUSCLE_PITCHES`, is much lower (~50% for `n_muscles=8`, ~85% for `n_muscles=95`).

**Fix plan.**

1. Rename the existing function to `biological_rate_ceiling` (semantic clarity).
2. Add `biological_pitch_ceiling(muscle_pitches, target_pitches)` that returns the fraction of Chopin pitches reachable by pitch-class match — reference impl in `wormuse_analytics.pipeline.reachable_pitches`.
3. Combined ceiling = `min(rate_ceiling, pitch_ceiling)` — what the worm can *actually* achieve under both rate and pitch constraints.
4. Update notebook 03 cell 24 summary table to report both.

**Snippet:**

```python
rate_ceil  = biological_rate_ceiling(p_celegans, t_on_chopin, window_s=DURATION, n_voices=8)
pitch_ceil = biological_pitch_ceiling(MUSCLE_PITCHES, chopin_pitches)
overall_ceil = min(rate_ceil['reachable_fraction'], pitch_ceil['reachable_fraction'])
print(f'Rate-only ceiling:   {rate_ceil["reachable_fraction"]:.3f}')
print(f'Pitch-only ceiling:  {pitch_ceil["reachable_fraction"]:.3f}')
print(f'Combined F1 ceiling: {2*overall_ceil/(overall_ceil+1):.3f}')
```

**Affected files.**
- `PyANNOW/src/pyannow/targets/midi_target.py` — rename + add the pitch ceiling.
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — cell 24 updated.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/tests/test_midi_target.py` — test the pitch ceiling on a known-pitch toy.
- `PyANNOW/TODO.md` — this entry.

**Connects to ISSUE-031** (the pitch bottleneck). The ceiling is the upper bound; ISSUE-031's fix (n_muscles=95) raises it.
