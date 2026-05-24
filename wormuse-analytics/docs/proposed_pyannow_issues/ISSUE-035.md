## ISSUE-035 — Pitch-aware F1 missing — onset-only metric ignores wrong notes `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P1 |
| **Severity** | Correctness — central scoring function does not measure the project goal |

**Description.** The wormuse goal is "teach a worm to play Chopin." Playing Chopin requires both the right **timing** AND the right **pitch**. Current scoring functions:

- `onset_loss(worm_onsets, chopin_onsets)` — ignores pitch (and is gameable, see ISSUE-016).
- `musical_f1(worm_onsets, chopin_onsets, tol_s=0.05)` — also ignores pitch.
- `ioi_similarity` — measures rhythmic distribution only.

A worm that gets the timing exactly right but plays random pitches scores **full marks**. That is not "Chopin-like." The metric the project actually needs:

**Definition (pitch-aware F1).** A worm onset at time t with pitch p is a true positive iff there exists a Chopin onset at time t' with pitch p' such that:

- `|t − t'| ≤ tol_s` (e.g. 50 ms), AND
- `p == p'` (exact pitch) OR `p mod 12 == p' mod 12` (same pitch class — pragmatic choice given the 8-muscle map covers only some pitches).

Matching is greedy in time order, no double-claims. Then F1 = harmonic mean of precision and recall.

**Reference implementation.** `wormuse_analytics.metrics.pitch_aware_f1` already does this.

**Fix plan.** Add to `pyannow/targets/midi_target.py`:

```python
def pitch_aware_f1(worm_onsets, worm_pitches, chopin_onsets, chopin_pitches,
                    tol_s=0.05, window_s=15.0, same_pitch_class=True) -> dict:
    """F1 requiring onsets to match in BOTH time and pitch."""
    # ... see wormuse_analytics.metrics.pitch_aware_f1
```

Notebook 03 cell 20 needs to be reorganised:

1. **Pitch-aware F1** becomes the LEFT panel (the headline metric).
2. **Plain F1@50ms** moves RIGHT — kept for backward comparison.
3. **`onset_loss`** is downgraded to a one-row table in the markdown summary, not a chart.

**Affected files.**
- `PyANNOW/src/pyannow/targets/midi_target.py` — add `pitch_aware_f1` + `velocity_correlation`.
- `PyANNOW/tests/test_midi_target.py` — tests (4-5 cases: empty / exact-match / pitch-mismatch / time-mismatch / partial-overlap).
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — cell 20 rework, cell 24 columns.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/TODO.md` — this entry.

**Connects to ISSUE-031 (pitch ceiling).** With `n_muscles=8`, the pitch-aware F1 ceiling is bounded; ISSUE-031 lifts that. Without ISSUE-031, pitch-aware F1 will appear "lower than expected"; the ceiling table in §Final of the audit notebook explains why.
