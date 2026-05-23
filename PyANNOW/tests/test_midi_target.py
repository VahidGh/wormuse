"""Tests for PyANNOW/src/pyannow/targets/midi_target.py

Covers:
  - MIDI file parsing (requires shared/examples/ asset — skips if absent)
  - onset_loss extremes (silence=1, exact=0)
  - Biological ceiling logic
  - Piano-roll shape correctness
  - note_rate_mismatch keys
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
