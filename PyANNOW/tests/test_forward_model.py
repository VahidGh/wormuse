"""Tests for PyANNOW/src/pyannow/composer/worm_optimizer_fast.py

Covers:
  - Output shapes and types for both n_muscles modes
  - Pitch map stays within piano range (MIDI 21-108)
  - Note events have valid timestamps, pitch indices, velocities
  - Deterministic with fixed seed
  - 95-cell model produces more pitches than 8-cell
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def result_8(default_params):
    from pyannow.composer.worm_optimizer_fast import run_forward_fast
    return run_forward_fast(
        default_params, duration_s=5.0, dt_ms=0.5,
        drive_freq_hz=0.4, drive_amplitude=8.0, random_seed=0, n_muscles=8)


@pytest.fixture(scope="module")
def result_95(default_params):
    from pyannow.composer.worm_optimizer_fast import run_forward_fast
    return run_forward_fast(
        default_params, duration_s=5.0, dt_ms=0.5,
        drive_freq_hz=0.4, drive_amplitude=8.0, random_seed=0, n_muscles=95)


class TestForwardOutputStructure:

    def test_result_keys_present(self, result_8):
        for key in ("t_arr_ms", "V_muscles", "note_onsets_s", "pitch_map", "n_muscles"):
            assert key in result_8, f"Missing key: {key}"

    def test_t_arr_length(self, result_8):
        N = int(5.0 * 1000 / 0.5)
        assert len(result_8["t_arr_ms"]) == N

    def test_V_muscles_shape_8(self, result_8):
        N = int(5.0 * 1000 / 0.5)
        assert result_8["V_muscles"].shape == (N, 8)

    def test_V_muscles_shape_95(self, result_95):
        N = int(5.0 * 1000 / 0.5)
        assert result_95["V_muscles"].shape == (N, 95)

    def test_V_muscles_finite(self, result_8):
        assert np.all(np.isfinite(result_8["V_muscles"])), "V_muscles must not contain NaN/Inf"

    def test_V_muscles_bounded(self, result_8):
        V = result_8["V_muscles"]
        assert V.min() >= -90.0 and V.max() <= 80.0, (
            f"V_muscles out of physiological range [{V.min():.1f}, {V.max():.1f}]")


class TestPitchMap:

    def test_pitch_map_length_8(self, result_8):
        assert len(result_8["pitch_map"]) == 8

    def test_pitch_map_length_95(self, result_95):
        assert len(result_95["pitch_map"]) == 95

    def test_pitch_map_within_piano_range(self, result_95):
        pm = result_95["pitch_map"]
        assert pm.min() >= 21, f"Lowest pitch {pm.min()} below piano minimum (21)"
        assert pm.max() <= 108, f"Highest pitch {pm.max()} above piano maximum (108)"

    def test_pitch_map_strictly_increasing(self, result_95):
        """Pitches should be ordered from bass to treble (head to tail = low to high)."""
        pm = result_95["pitch_map"]
        assert np.all(np.diff(pm) >= 0), "Pitch map must be non-decreasing"

    def test_95_more_pitches_than_8(self, result_8, result_95):
        p8  = set(result_8["pitch_map"])
        p95 = set(result_95["pitch_map"])
        assert len(p95) > len(p8), "95-cell model must cover more pitches than 8-cell"


class TestNoteEvents:

    def test_note_onsets_list_type(self, result_8):
        assert isinstance(result_8["note_onsets_s"], list)

    def test_note_event_tuple_structure(self, result_8):
        for ev in result_8["note_onsets_s"]:
            assert len(ev) == 3, "Each note event must be (time_s, muscle_idx, velocity)"
            t_s, idx, vel = ev
            assert t_s >= 0.0, "Note onset time must be non-negative"
            assert t_s <= 5.0 + 1e-3, "Note onset must be within simulation duration"
            assert 0 <= idx < 8, f"Muscle index {idx} out of range for 8-cell model"
            assert 1 <= vel <= 127, f"Velocity {vel} out of MIDI range"

    def test_note_events_sorted(self, result_8):
        times = [ev[0] for ev in result_8["note_onsets_s"]]
        assert times == sorted(times), "Note events must be sorted by onset time"


class TestDeterminism:

    def test_same_seed_same_output(self, default_params):
        from pyannow.composer.worm_optimizer_fast import run_forward_fast
        r1 = run_forward_fast(default_params, duration_s=3.0, dt_ms=0.5,
                               drive_freq_hz=0.4, drive_amplitude=8.0,
                               random_seed=7, n_muscles=8)
        r2 = run_forward_fast(default_params, duration_s=3.0, dt_ms=0.5,
                               drive_freq_hz=0.4, drive_amplitude=8.0,
                               random_seed=7, n_muscles=8)
        np.testing.assert_array_equal(r1["V_muscles"], r2["V_muscles"],
                                       err_msg="Same seed must give identical V_muscles")

    def test_different_seeds_different_output(self, default_params):
        from pyannow.composer.worm_optimizer_fast import run_forward_fast
        r1 = run_forward_fast(default_params, duration_s=3.0, dt_ms=0.5,
                               drive_freq_hz=0.4, drive_amplitude=8.0,
                               random_seed=1, n_muscles=8)
        r2 = run_forward_fast(default_params, duration_s=3.0, dt_ms=0.5,
                               drive_freq_hz=0.4, drive_amplitude=8.0,
                               random_seed=2, n_muscles=8)
        assert not np.array_equal(r1["V_muscles"], r2["V_muscles"]), (
            "Different seeds should give different muscle traces")


class TestPitchGenerator:

    def test_generate_8_returns_legacy(self):
        from pyannow.composer.worm_optimizer import generate_muscle_pitches
        p8 = generate_muscle_pitches(8)
        assert list(p8) == [61, 64, 66, 68, 71, 73, 76, 78], (
            "8-cell must return the C# minor pentatonic legacy array")

    def test_generate_95_length(self):
        from pyannow.composer.worm_optimizer import generate_muscle_pitches
        p95 = generate_muscle_pitches(95)
        assert len(p95) == 95

    def test_generate_95_in_range(self):
        from pyannow.composer.worm_optimizer import generate_muscle_pitches
        p95 = generate_muscle_pitches(95)
        assert p95.min() >= 21 and p95.max() <= 108, (
            f"95-cell pitches out of piano range: [{p95.min()}, {p95.max()}]")

    @pytest.mark.parametrize("n", [1, 4, 12, 50, 88, 95])
    def test_generate_arbitrary_n(self, n):
        from pyannow.composer.worm_optimizer import generate_muscle_pitches
        p = generate_muscle_pitches(n)
        assert len(p) == n
        assert p.min() >= 21 and p.max() <= 108
