"""Tests for PyANNOW/src/pyannow/targets/midi_target.py

Covers:
  - MIDI file parsing (requires shared/examples/ asset — skips if absent)
  - onset_loss extremes (silence=1, exact=0)
  - Biological ceiling logic
  - Piano-roll shape correctness
  - note_rate_mismatch keys
  - pitch_aware_f1 (ISSUE-035, v0.8.0)
  - biological_pitch_ceiling (ISSUE-037, v0.8.0)
  - precision_recall_at_tolerances (ISSUE-020, v0.8.0)
  - bootstrap_musical_f1 (ISSUE-021, v0.8.0)
"""
from __future__ import annotations

import numpy as np
import pytest


class TestOnsetLoss:

    def test_silence_gives_max_loss(self):
        """Empty worm onset array vs any target should give loss = 1.0."""
        from pyannow.targets.midi_target import onset_loss
        target = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        L = onset_loss(np.array([]), target, window_s=5.0)
        assert L == pytest.approx(1.0, abs=1e-6), f"Empty onsets → loss must be 1.0 (got {L})"

    def test_exact_match_gives_near_zero_loss(self):
        """Identical onset arrays should produce near-zero loss."""
        from pyannow.targets.midi_target import onset_loss
        onsets = np.linspace(0, 4.8, 30)
        L = onset_loss(onsets, onsets, window_s=5.0)
        assert L < 0.01, f"Exact match should give near-0 loss, got {L:.5f}"

    def test_loss_between_zero_and_one(self):
        """Loss must always be in [0, 1]."""
        from pyannow.targets.midi_target import onset_loss
        rng = np.random.default_rng(0)
        for _ in range(10):
            w = rng.uniform(0, 5.0, 20)
            t = rng.uniform(0, 5.0, 25)
            L = onset_loss(np.sort(w), np.sort(t), window_s=5.0)
            assert 0.0 <= L <= 1.0, f"loss={L} out of [0,1]"

    def test_loss_worsens_with_offset(self):
        """Shifting all worm onsets by 1s should increase the loss."""
        from pyannow.targets.midi_target import onset_loss
        target = np.linspace(0, 4, 20)
        L_good = onset_loss(target, target, window_s=5.0)
        L_bad  = onset_loss(target + 1.0, target, window_s=5.0)
        assert L_bad > L_good, "Shifted onsets should give worse loss"


class TestBiologicalCeiling:

    def test_ceiling_fraction_in_01(self, default_params):
        """Ceiling fraction must be in [0, 1]."""
        from pyannow.targets.midi_target import biological_ceiling
        target = np.linspace(0, 30, 100)   # 100 notes in 30s = 3.3/s
        result = biological_ceiling(default_params, target, window_s=30.0)
        frac = result["reachable_fraction"]
        assert 0.0 <= frac <= 1.0, f"ceiling fraction={frac} out of [0,1]"

    def test_slow_piece_fully_reachable(self, default_params):
        """Notes spaced 5s apart must all be reachable (tau_refrac << 5s)."""
        from pyannow.targets.midi_target import biological_ceiling
        sparse_target = np.arange(0, 30, 5.0)   # 1 note every 5 seconds
        result = biological_ceiling(default_params, sparse_target, window_s=30.0)
        assert result["reachable_fraction"] == pytest.approx(1.0, abs=0.01), (
            "Notes 5s apart must all be reachable given tau_refrac ~65ms")

    def test_dense_piece_partial_ceiling_single_voice(self, default_params):
        """Single-voice ceiling: notes 30ms apart must be < 1 (below tau_refrac ~65ms)."""
        from pyannow.targets.midi_target import biological_ceiling
        dense_target = np.arange(0, 5.0, 0.030)   # note every 30ms
        result = biological_ceiling(default_params, dense_target, window_s=5.0, n_voices=1)
        assert result["reachable_fraction"] < 1.0, (
            "Single-voice: notes 30ms apart should hit the refractory ceiling (tau_refrac ~65ms)")

    def test_dense_piece_nvoice_fully_reachable(self, default_params):
        """95-voice ceiling: notes 30ms apart must all be reachable (95 voices >> needed)."""
        from pyannow.targets.midi_target import biological_ceiling
        dense_target = np.arange(0, 5.0, 0.030)   # note every 30ms = 33 notes/s
        result = biological_ceiling(default_params, dense_target, window_s=5.0, n_voices=95)
        assert result["reachable_fraction"] == 1.0, (
            "95 voices at 33 notes/s is well within capacity (95 × 15 notes/s max)")


class TestMidiParsing:

    def test_parse_returns_events_and_bpm(self, midi_path):
        from pyannow.targets.midi_target import parse_midi
        events, bpm = parse_midi(midi_path)
        assert len(events) > 0, "Parsed MIDI should contain note events"
        assert bpm > 0, "BPM must be positive"

    def test_note_onsets_sorted(self, midi_path):
        from pyannow.targets.midi_target import parse_midi, note_onsets
        events, _ = parse_midi(midi_path)
        t = note_onsets(events)
        assert np.all(np.diff(t) >= 0), "Note onsets must be non-decreasing"

    def test_pitch_range_valid(self, midi_path):
        from pyannow.targets.midi_target import parse_midi
        events, _ = parse_midi(midi_path)
        pitches = [e.pitch for e in events]
        assert all(21 <= p <= 108 for p in pitches), (
            "All pitches must be within the 88-key piano range (21-108)")

    def test_velocity_range_valid(self, midi_path):
        from pyannow.targets.midi_target import parse_midi
        events, _ = parse_midi(midi_path)
        velocities = [e.velocity for e in events]
        assert all(1 <= v <= 127 for v in velocities), "Velocities must be in [1,127]"

    def test_note_rate_mismatch_keys(self, midi_path):
        from pyannow.targets.midi_target import parse_midi, note_onsets, note_rate_mismatch
        events, _ = parse_midi(midi_path)
        t = note_onsets(events)
        result = note_rate_mismatch(t[:20], t[:20], window_s=t[19])
        assert "worm_rate_Hz" in result
        assert "target_rate_Hz" in result
        assert "worm_N" in result


class TestPianoRoll:

    def test_piano_roll_shape(self, midi_path):
        from pyannow.targets.midi_target import parse_midi, piano_roll
        events, _ = parse_midi(midi_path)
        pitches, times, roll = piano_roll(events, resolution_s=0.05, clip_s=10.0)
        assert roll.shape == (len(pitches), len(times)), "roll.shape must be (n_pitches, n_times)"
        assert roll.dtype == bool, "Piano roll must be boolean"

class TestMusicalMetrics:

    def test_musical_f1_perfect_match(self):
        """F1 must be 1.0 when worm onsets == target onsets."""
        from pyannow.targets.midi_target import musical_f1
        onsets = np.linspace(0.1, 14.9, 40)
        result = musical_f1(onsets, onsets, tol_s=0.05, window_s=15.0)
        assert result["f1"] == pytest.approx(1.0, abs=1e-6), (
            f"Identical arrays must give F1=1.0, got {result['f1']:.4f}")

    def test_musical_f1_single_note_zero_recall(self):
        """1-note worm vs 40-note target must give F1 ≈ 0 (sparse → recall≈0)."""
        from pyannow.targets.midi_target import musical_f1
        target = np.linspace(0.1, 14.9, 40)
        worm   = np.array([7.0])
        result = musical_f1(worm, target, tol_s=0.05, window_s=15.0)
        assert result["f1"] < 0.1, (
            f"Single-note worm should give F1<0.1 vs 40 Chopin notes, got {result['f1']:.4f}")
        assert result["recall"] < 0.1, "Recall must be near-zero for single-note worm"

    def test_ioi_similarity_identical(self):
        """ioi_similarity must return 1.0 for identical onset arrays."""
        from pyannow.targets.midi_target import ioi_similarity
        onsets = np.linspace(0.2, 14.8, 40)
        sim = ioi_similarity(onsets, onsets, window_s=15.0)
        assert sim == pytest.approx(1.0, abs=0.02), (
            f"Identical IOI distributions must give similarity≈1.0, got {sim:.4f}")

    def test_ioi_similarity_orthogonal(self):
        """Non-overlapping IOI distributions must give near-zero similarity."""
        from pyannow.targets.midi_target import ioi_similarity
        # Worm: IOIs all at 2.0s (very slow) — falls outside [0, 2) histogram range
        worm   = np.arange(0.0, 15.0, 2.0)
        # Target: IOIs all at ~0.1s (fast) — spike near 0.1 in histogram
        target = np.arange(0.0, 15.0, 0.1)
        sim = ioi_similarity(worm, target, window_s=15.0)
        assert sim < 0.20, (
            f"Orthogonal IOI distributions should give near-zero similarity, got {sim:.4f}")


class TestPianoRoll:

    def test_piano_roll_no_notes_before_onset(self, midi_path):
        """Piano roll first bin must not be unusually dense (opening chord check).

        The Nocturne Op.posth in C# minor opens with a 4-note chord at t≈0.
        Previous bound of 2 was written for the old Raindrop MIDI and is wrong.
        The correct bound is the observed opening chord size of 4.
        """
        from pyannow.targets.midi_target import parse_midi, piano_roll
        events, _ = parse_midi(midi_path)
        _, _, roll = piano_roll(events, resolution_s=0.05, clip_s=10.0)
        # Nocturne opens with a 4-note chord; allow up to 5 to be safe
        n_first = int(roll[:, 0].sum())
        assert n_first <= 5, (
            f"At most 5 notes active in first time bin (Nocturne opens with 4-note chord); "
            f"got {n_first} — MIDI may have changed")


# ─────────────────────────────────────────────────────────────────────────────
# v0.8.0 — Pitch-aware metrics  (ISSUE-035, ISSUE-037, ISSUE-020, ISSUE-021)
# ─────────────────────────────────────────────────────────────────────────────

class TestPitchAwareF1:
    """Tests for pitch_aware_f1() — requires BOTH time AND pitch match."""

    def test_exact_match_gives_f1_one(self):
        """Perfect time+pitch match → F1 = 1.0."""
        from pyannow.targets.midi_target import pitch_aware_f1
        onsets  = np.linspace(0.5, 9.5, 20)
        pitches = np.tile([60, 62, 64, 65, 67], 4)   # 20 notes
        r = pitch_aware_f1(onsets, pitches, onsets, pitches)
        assert r["f1"] == pytest.approx(1.0, abs=1e-6), (
            f"Identical time+pitch arrays must give F1=1.0, got {r['f1']:.4f}")

    def test_pitch_mismatch_gives_zero_f1(self):
        """Correct timing but completely wrong pitches → F1 = 0.0."""
        from pyannow.targets.midi_target import pitch_aware_f1
        onsets   = np.linspace(0.5, 9.5, 20)
        pitches_w = np.full(20, 60)   # worm: all C4
        pitches_c = np.full(20, 61)   # Chopin: all C#4
        r = pitch_aware_f1(onsets, pitches_w, onsets, pitches_c,
                           same_pitch_class=False)
        assert r["f1"] == pytest.approx(0.0, abs=1e-6), (
            f"Wrong pitches (no class match) must give F1=0.0, got {r['f1']:.4f}")

    def test_pitch_class_match_scores_correctly(self):
        """Pitch-class match (same note, different octave) counts with same_pitch_class=True."""
        from pyannow.targets.midi_target import pitch_aware_f1
        onsets = np.array([1.0, 2.0, 3.0])
        p_worm   = np.array([60, 62, 64])       # C4, D4, E4
        p_chopin = np.array([72, 74, 76])       # C5, D5, E5 (same pitch classes)
        r = pitch_aware_f1(onsets, p_worm, onsets, p_chopin,
                           same_pitch_class=True)
        assert r["f1"] == pytest.approx(1.0, abs=1e-6), (
            f"Same pitch-class different octave must match, got F1={r['f1']:.4f}")

    def test_empty_worm_gives_zero(self):
        """Empty worm onsets → F1 = 0."""
        from pyannow.targets.midi_target import pitch_aware_f1
        r = pitch_aware_f1(
            np.array([]), np.array([]),
            np.linspace(0.5, 9.5, 10), np.full(10, 60),
        )
        assert r["f1"] == 0.0

    def test_pitch_acc_diagnostic(self):
        """pitch_acc must be < 1 when some timing matches have wrong pitch."""
        from pyannow.targets.midi_target import pitch_aware_f1
        onsets = np.array([1.0, 2.0, 3.0, 4.0])
        p_worm   = np.array([60, 99, 99, 60])    # 2 correct, 2 wrong pitch
        p_chopin = np.array([60, 60, 60, 60])
        r = pitch_aware_f1(onsets, p_worm, onsets, p_chopin,
                           same_pitch_class=False)
        assert 0.0 < r["pitch_acc"] < 1.0, (
            f"pitch_acc should be partial, got {r['pitch_acc']:.3f}")


class TestBiologicalPitchCeiling:
    """Tests for biological_pitch_ceiling() — ISSUE-037."""

    def test_96cell_covers_all_pitch_classes(self):
        """96-cell Boyle model (MIDI 24-119) covers all 12 pitch classes."""
        from pyannow.targets.midi_target import biological_pitch_ceiling
        # MUSCLE_PITCHES_96 = chromatic MIDI 24-119 (all 12 pitch classes × 8 octaves)
        # Replicate the 96-cell pitch map without importing worm_optimizer (no scipy dep)
        muscle_pitches_96 = np.concatenate([
            np.arange(24, 48),   # DL: C1-B2
            np.arange(48, 72),   # VL: C3-B4
            np.arange(72, 96),   # DR: C5-B6
            np.arange(96, 120),  # VR: C7-B8
        ]).astype(int)
        # Chopin pitches spanning all 12 pitch classes
        chopin_pitches = np.arange(48, 84)   # C3-B5 chromatic (36 notes)
        r = biological_pitch_ceiling(muscle_pitches_96, chopin_pitches,
                                     same_pitch_class=True)
        assert r["reachable_fraction"] == pytest.approx(1.0, abs=1e-6), (
            f"96-cell model must reach all pitch classes, got {r['reachable_fraction']:.4f}")
        assert r["unique_reachable_classes"] == 12

    def test_8cell_limited_coverage(self):
        """8-cell C#m model covers fewer pitch classes than Chopin needs."""
        from pyannow.targets.midi_target import biological_pitch_ceiling
        # C#m pentatonic (8 cells): C#4, D#4, E4, F#4, G#4, A4, B4, C#5
        # = MIDI [61, 64, 66, 68, 71, 73, 76, 78]
        muscle_pitches_8 = np.array([61, 64, 66, 68, 71, 73, 76, 78])
        # Chopin pitches with many non-C#m notes (full chromatic octave)
        chopin_pitches = np.arange(60, 72)   # C4-B4 chromatic (12 notes)
        r = biological_pitch_ceiling(muscle_pitches_8, chopin_pitches,
                                     same_pitch_class=True)
        # C#m has 7 pitch classes; chromatic has 12; so ceiling < 1.0
        assert r["reachable_fraction"] < 1.0, (
            "8-cell C#m model must NOT reach all pitch classes of a chromatic scale")

    def test_output_keys(self):
        """All expected keys must be present in the return dict."""
        from pyannow.targets.midi_target import biological_pitch_ceiling
        r = biological_pitch_ceiling(np.array([60, 62, 64]), np.array([60, 61, 62, 63]))
        for key in ("reachable_fraction", "reachable_N", "total_N",
                    "unique_reachable_classes", "unique_target_classes", "n_muscles"):
            assert key in r, f"Missing key '{key}' in biological_pitch_ceiling result"


class TestPrecisionRecallAtTolerances:
    """Tests for precision_recall_at_tolerances() — ISSUE-020."""

    def test_returns_one_entry_per_tolerance(self):
        from pyannow.targets.midi_target import precision_recall_at_tolerances
        onsets = np.linspace(0.1, 9.9, 20)
        tols   = (0.025, 0.05, 0.10, 0.20)
        results = precision_recall_at_tolerances(onsets, onsets, tols=tols)
        assert len(results) == len(tols), (
            f"Expected {len(tols)} entries, got {len(results)}")

    def test_f1_monotone_in_tolerance(self):
        """F1 must be non-decreasing as tolerance increases (exact match onsets)."""
        from pyannow.targets.midi_target import precision_recall_at_tolerances
        # Worm onsets shifted by +30ms relative to Chopin
        chopin = np.linspace(0.5, 9.5, 20)
        worm   = chopin + 0.03
        tols   = (0.02, 0.04, 0.05, 0.10)
        results = precision_recall_at_tolerances(worm, chopin, tols=tols)
        f1s = [r["f1"] for r in results]
        for i in range(len(f1s) - 1):
            assert f1s[i] <= f1s[i + 1] + 1e-9, (
                f"F1 must not decrease with wider tolerance: {f1s}")

    def test_perfect_match_gives_f1_one_at_all_tols(self):
        from pyannow.targets.midi_target import precision_recall_at_tolerances
        onsets  = np.linspace(0.5, 9.5, 20)
        results = precision_recall_at_tolerances(onsets, onsets)
        for r in results:
            assert r["f1"] == pytest.approx(1.0, abs=1e-6), (
                f"Perfect match must give F1=1.0 at tol={r['tol_s']}, got {r['f1']:.4f}")


class TestBootstrapMusicalF1:
    """Tests for bootstrap_musical_f1() — ISSUE-021."""

    def test_ci_contains_mean(self):
        """Bootstrap CI must contain the mean."""
        from pyannow.targets.midi_target import bootstrap_musical_f1
        onsets = np.linspace(0.5, 9.5, 20)
        r = bootstrap_musical_f1(onsets, onsets, n_boot=100, seed=42)
        assert r["ci_low"] <= r["mean_f1"] <= r["ci_high"], (
            f"Mean {r['mean_f1']:.4f} not in CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")

    def test_perfect_match_high_mean(self):
        """Bootstrap of identical arrays must give mean F1 well above zero.

        Note: bootstrapping resamples target onsets with replacement, so the
        resampled target may have duplicates/gaps vs. the fixed worm onsets.
        Mean F1 is typically 0.65-0.80 for identical inputs — not 1.0.
        The key assertion is that the mean is substantially above a random baseline.
        """
        from pyannow.targets.midi_target import bootstrap_musical_f1
        onsets = np.linspace(0.5, 9.5, 30)
        r = bootstrap_musical_f1(onsets, onsets, n_boot=100, seed=0)
        assert r["mean_f1"] > 0.50, (
            f"Perfect-match bootstrap mean must be >0.50, got {r['mean_f1']:.4f}")

    def test_empty_gives_zeros(self):
        from pyannow.targets.midi_target import bootstrap_musical_f1
        r = bootstrap_musical_f1(np.array([]), np.array([1.0, 2.0]), n_boot=10)
        assert r["mean_f1"] == 0.0 and r["ci_low"] == 0.0
