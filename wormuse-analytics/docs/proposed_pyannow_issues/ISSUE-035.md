## ISSUE-035 — Pitch-aware F1 missing — onset-only metric ignores wrong notes `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | ✅ Resolved — v0.8.0 |
| **Priority** | P1 |
| **Severity** | Correctness — logic problem #7 — central scoring function does not measure the project goal |

**Description.** `musical_f1()` matches onset times but ignores pitch. A worm hitting all the right timings with random pitches scores full marks. That is not "playing Chopin." With the 96-cell Boyle model now covering all 12 pitch classes (ISSUE-031), pitch-aware evaluation is meaningful.

**Fix (v0.8.0).** `pitch_aware_f1()` in `midi_target.py`:

```python
def pitch_aware_f1(worm_onsets, worm_pitches, chopin_onsets, chopin_pitches,
                    tol_s=0.05, window_s=15.0, same_pitch_class=True) -> dict:
    """F1 requiring BOTH time (±tol_s) AND pitch (exact or pitch-class) match.
    Greedy one-to-one matching; no double-claims.
    Returns: f1, precision, recall, tp, n_worm, n_chopin, pitch_acc
    """
```

Key design choices:
- `same_pitch_class=True` (default): pitch-class matching (p mod 12) gives credit for correct note in wrong octave — pragmatic given the 8-octave Boyle layout.
- `pitch_acc` diagnostic: among timing matches, fraction with correct pitch — isolates pitch error from timing error.
- Greedy matching in time order ensures no note is double-claimed.

**Score improvement from 96-cell model.** With 8-cell model: pitch ceiling ≈ 40% → pitch_aware_f1 << musical_f1. With 96-cell model: pitch ceiling = 100% → pitch_aware_f1 bounded only by timing + mapping quality.

**Tested.** `test_midi_target.py::TestPitchAwareF1` — 5 cases:
- exact match → F1=1.0
- pitch mismatch → F1=0.0 (same_pitch_class=False)
- pitch-class match (different octave) → F1=1.0
- empty worm → F1=0.0
- partial match → pitch_acc diagnostic verified

**AppStat connection.** L05-L06: F1 = harmonic mean of precision and recall. Pitch-aware variant extends the standard MIR evaluation to the multi-modal (time × pitch) setting.

**Category:** `Category A — Metrics & Scoring`
