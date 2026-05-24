"""Tests for PyANNOW/src/pyannow/composer/worm_optimizer_fast.py

Covers:
  - Output shapes and types for n_muscles=8, 95, 96
  - Pitch map stays within piano range (MIDI 21-108) for legacy models
  - 96-cell Boyle model: 4-quadrant chromatic range MIDI 24-119
  - Note events have valid timestamps, pitch indices, velocities
  - Deterministic with fixed seed
  - generate_neural_activity_302: shape, rank, dtype
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


@pytest.fixture(scope="module")
def result_96(default_params):
    from pyannow.composer.worm_optimizer_fast import run_forward_fast
    return run_forward_fast(
        default_params, duration_s=5.0, dt_ms=0.5,
        drive_freq_hz=1.5, drive_amplitude=12.0, random_seed=0, n_muscles=96)


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

    def test_generate_96_length(self):
        from pyannow.composer.worm_optimizer import generate_muscle_pitches
        p96 = generate_muscle_pitches(96)
        assert len(p96) == 96, "96-cell Boyle model must return 96 pitches"

    def test_generate_96_quadrant_ranges(self):
        from pyannow.composer.worm_optimizer import generate_muscle_pitches
        p96 = generate_muscle_pitches(96)
        # DL quadrant (0-23): C1-B2 = MIDI 24-47
        assert p96[:24].min() == 24 and p96[:24].max() == 47, "DL quadrant MIDI 24-47"
        # VL quadrant (24-47): C3-B4 = MIDI 48-71
        assert p96[24:48].min() == 48 and p96[24:48].max() == 71, "VL quadrant MIDI 48-71"
        # DR quadrant (48-71): C5-B6 = MIDI 72-95
        assert p96[48:72].min() == 72 and p96[48:72].max() == 95, "DR quadrant MIDI 72-95"
        # VR quadrant (72-95): C7-B8 = MIDI 96-119
        assert p96[72:96].min() == 96 and p96[72:96].max() == 119, "VR quadrant MIDI 96-119"

    def test_generate_96_all_unique(self):
        from pyannow.composer.worm_optimizer import generate_muscle_pitches
        p96 = generate_muscle_pitches(96)
        assert len(set(p96)) == 96, "96-cell model must have 96 unique chromatic pitches"

    @pytest.mark.parametrize("n", [1, 4, 12, 50, 88, 95])
    def test_generate_arbitrary_n(self, n):
        from pyannow.composer.worm_optimizer import generate_muscle_pitches
        p = generate_muscle_pitches(n)
        assert len(p) == n
        assert p.min() >= 21 and p.max() <= 108


class TestNeuralActivity302:

    def test_shape(self):
        from pyannow.composer.worm_optimizer import generate_neural_activity_302
        X = generate_neural_activity_302(n_steps=1000, dt_ms=0.5)
        assert X.shape == (302, 1000), f"Expected (302, 1000), got {X.shape}"

    def test_dtype(self):
        from pyannow.composer.worm_optimizer import generate_neural_activity_302
        X = generate_neural_activity_302(n_steps=500)
        assert X.dtype == np.float32

    def test_rank_geq_4(self):
        from pyannow.composer.worm_optimizer import generate_neural_activity_302
        X = generate_neural_activity_302(n_steps=2000, seed=42)
        # The matrix must have at least 4 singular values explaining ≥ 10% of variance each
        s = np.linalg.svd(X, compute_uv=False)
        # At minimum: 4 PCs with singular value > 10% of the largest
        n_significant = (s > s[0] * 0.1).sum()
        assert n_significant >= 4, (
            f"Need ≥ 4 significant PCs for NAML steps to outperform Step 0; got {n_significant}")

    def test_finite(self):
        from pyannow.composer.worm_optimizer import generate_neural_activity_302
        X = generate_neural_activity_302(n_steps=500)
        assert np.all(np.isfinite(X)), "X_neural must not contain NaN/Inf"

    def test_deterministic(self):
        from pyannow.composer.worm_optimizer import generate_neural_activity_302
        X1 = generate_neural_activity_302(n_steps=500, seed=7)
        X2 = generate_neural_activity_302(n_steps=500, seed=7)
        np.testing.assert_array_equal(X1, X2)

    def test_different_seeds_different(self):
        from pyannow.composer.worm_optimizer import generate_neural_activity_302
        X1 = generate_neural_activity_302(n_steps=500, seed=1)
        X2 = generate_neural_activity_302(n_steps=500, seed=2)
        assert not np.array_equal(X1, X2)


class TestBoyle96CellForward:

    def test_shape_96(self, result_96):
        N = int(5.0 * 1000 / 0.5)
        assert result_96["V_muscles"].shape == (N, 96), (
            f"96-cell model must have V_muscles shape ({N}, 96)")

    def test_pitch_map_96(self, result_96):
        assert len(result_96["pitch_map"]) == 96

    def test_pitch_quadrant_structure(self, result_96):
        pm = result_96["pitch_map"]
        assert pm[:24].min() == 24,  "DL quadrant must start at MIDI 24 (C1)"
        assert pm[24:48].min() == 48, "VL quadrant must start at MIDI 48 (C3)"
        assert pm[48:72].min() == 72, "DR quadrant must start at MIDI 72 (C5)"
        assert pm[72:].min() == 96,   "VR quadrant must start at MIDI 96 (C7)"

    def test_n_muscles_key(self, result_96):
        assert result_96["n_muscles"] == 96

    def test_96_more_unique_pitches_than_8(self, result_8, result_96):
        p8  = set(result_8["pitch_map"])
        p96 = set(result_96["pitch_map"])
        assert len(p96) > len(p8)

    def test_default_is_96(self, default_params):
        from pyannow.composer.worm_optimizer_fast import run_forward_fast
        r = run_forward_fast(default_params, duration_s=2.0, dt_ms=0.5,
                              drive_freq_hz=1.5, drive_amplitude=12.0, random_seed=0)
        assert r["n_muscles"] == 96, "Default n_muscles must be 96 (Boyle 4×24 architecture)"
