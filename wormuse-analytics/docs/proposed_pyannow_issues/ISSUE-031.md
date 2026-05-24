## ISSUE-031 — 8-muscle pitch bottleneck caps pitch-aware F1 at ~40% `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | ✅ Resolved — commit `1dbd3f7` (v0.7.0, part of ISSUE-018) |
| **Priority** | P1 |
| **Severity** | Architecture correctness — logic problem #3 |

**Description.** Original 8-cell C#m model covers only 7 of 12 pitch classes. Chopin's Nocturne uses all 12. Pitch-aware F1 ceiling was ≈40% regardless of timing quality.

**Resolution (v0.7.0).**
- **96-cell Boyle 4×24 model** is now the default: chromatic MIDI 24-119 (DL→C1-B2, VL→C3-B4, DR→C5-B6, VR→C7-B8).
- All 12 pitch classes covered → pitch ceiling = 100% (confirmed in `test_midi_target.py::TestBiologicalPitchCeiling::test_96cell_covers_all_pitch_classes`).
- `MUSCLE_PITCHES_96` constant; `biological_pitch_ceiling()` confirms coverage.
- 8-cell legacy kept for backward compatibility (`MUSCLE_PITCHES`, `generate_muscle_pitches(8)`).

**Measured impact.** `biological_pitch_ceiling(MUSCLE_PITCHES_96, chopin_pitches)` = 1.000 (was 0.400 with 8-cell model). The F1 ceiling is now determined by timing quality and rhythmic structure, not pitch coverage.

**Category:** `Category D — Architecture & Data Pipeline`
