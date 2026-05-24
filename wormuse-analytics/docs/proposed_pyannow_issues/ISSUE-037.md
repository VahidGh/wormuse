## ISSUE-037 — "Biological ceiling" formula misses pitch dimension `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | ✅ Resolved — v0.8.0 (rate ceiling fixed ISSUE-002; pitch ceiling added v0.8.0) |
| **Priority** | P2 |
| **Severity** | Correctness — ceiling overstated; logic problem #9 |

**Description.** Two versions of the ceiling were confused:
- **Rate ceiling** (`biological_ceiling()`): already fixed in ISSUE-002 (n-voice greedy, ~100% for 96 cells).
- **Pitch ceiling** (missing until v0.8.0): what fraction of Chopin's pitches can the worm physically produce given its fixed muscle-pitch map?

With the 8-cell model, pitch ceiling ≈ 40% (7 of 12 pitch classes). With the 96-cell Boyle model, pitch ceiling = 100%. The combined ceiling = `min(rate_ceiling, pitch_ceiling)`.

**Fix (v0.8.0).** `biological_pitch_ceiling(muscle_pitches, target_pitches, same_pitch_class=True)` added to `midi_target.py`:

```python
rate_ceil  = biological_ceiling(p_celegans, t_on_chopin, n_voices=96)
pitch_ceil = biological_pitch_ceiling(MUSCLE_PITCHES_96, chopin_pitches)
overall    = min(rate_ceil["reachable_fraction"], pitch_ceil["reachable_fraction"])
print(f"Rate ceiling:  {rate_ceil['reachable_fraction']:.3f}")
print(f"Pitch ceiling: {pitch_ceil['reachable_fraction']:.3f}")   # 1.000 with 96-cell model
print(f"Combined:      {overall:.3f}")
```

**Measured result (96-cell model):**
- Rate ceiling: 1.000 (96 voices easily handle Chopin's 4.4 notes/s)
- Pitch ceiling: 1.000 (all 12 pitch classes covered, MIDI 24-119)
- Combined F1 ceiling: 1.000 (the bottleneck is now purely the learning algorithm)

**Tested.** `test_midi_target.py::TestBiologicalPitchCeiling` — 3 tests confirming 96-cell full coverage and 8-cell partial coverage.

**Category:** `Category A — Metrics & Scoring`
