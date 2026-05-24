"""Tests for pyannow.step1_svd.procrustes — v0.8.0 additions.

Covers (ISSUE-034, ISSUE-032):
  - build_chopin_features() auto-k selection by variance threshold
  - build_chopin_features() explicit k_chopin
  - chopin_cumvar() diagnostic
  - procrustes_align() standardize=True vs False
  - procrustes_align() return keys and shapes
"""
from __future__ import annotations

import numpy as np
import pytest


# ─── Minimal synthetic NoteEvent ──────────────────────────────────────────────

class _Ev:
    """Lightweight stand-in for NoteEvent (avoids mido dependency in these tests)."""
    def __init__(self, t, pitch, vel=80, dur=0.2):
        self.time_s   = t
        self.pitch    = pitch
        self.velocity = vel
        self.duration = dur


def _make_events(n=40, duration_s=10.0, seed=7):
    """Synthetic event list spanning several pitch classes."""
    rng  = np.random.default_rng(seed)
    times   = np.sort(rng.uniform(0.05, duration_s - 0.1, n))
    pitches = rng.integers(48, 84, size=n)  # C3–B5 (many pitch classes)
    return [_Ev(t, int(p)) for t, p in zip(times, pitches)]


# ─── build_chopin_features ────────────────────────────────────────────────────

class TestBuildChopinFeatures:

    def test_explicit_k_chopin_controls_columns(self):
        """Explicit k_chopin=8 must produce exactly 8 columns."""
        from pyannow.step1_svd.procrustes import build_chopin_features
        events = _make_events(40)
        C = build_chopin_features(events, duration_s=10.0, n_bins=100, k_chopin=8)
        assert C.shape == (100, 8), f"Expected (100, 8), got {C.shape}"

    def test_auto_k_selects_by_variance(self):
        """Auto-k (k_chopin=None) must pick k such that cumvar ≥ 0.90."""
        from pyannow.step1_svd.procrustes import build_chopin_features, chopin_cumvar
        events = _make_events(40)
        C = build_chopin_features(events, duration_s=10.0, n_bins=100,
                                  k_chopin=None, var_threshold=0.90)
        k_auto = C.shape[1]
        cv = chopin_cumvar(events, duration_s=10.0, n_bins=100)
        # k_auto should satisfy cumvar[k_auto-1] >= 0.90
        assert cv[k_auto - 1] >= 0.90 - 1e-6, (
            f"Auto-k={k_auto} cumvar={cv[k_auto-1]:.4f} < 0.90 threshold")

    def test_auto_k_not_unnecessarily_large(self):
        """Auto-k should not overshoot (k must be < number of pitches)."""
        from pyannow.step1_svd.procrustes import build_chopin_features
        events = _make_events(20, seed=42)
        C = build_chopin_features(events, duration_s=10.0, n_bins=100,
                                  k_chopin=None, var_threshold=0.90)
        # k must be at most the number of unique pitches
        n_pitches = len({e.pitch for e in events})
        assert C.shape[1] <= n_pitches, (
            f"Auto-k={C.shape[1]} exceeds number of unique pitches={n_pitches}")

    def test_empty_events_returns_zeros(self):
        """Empty event list must return a zero matrix (default k=8 fallback)."""
        from pyannow.step1_svd.procrustes import build_chopin_features
        C = build_chopin_features([], duration_s=10.0, n_bins=100, k_chopin=8)
        assert C.shape == (100, 8)
        assert np.all(C == 0.0)

    def test_output_dtype_is_float(self):
        from pyannow.step1_svd.procrustes import build_chopin_features
        events = _make_events(30)
        C = build_chopin_features(events, duration_s=10.0, n_bins=100)
        assert C.dtype in (np.float32, np.float64), f"Expected float, got {C.dtype}"


class TestChopinCumvar:

    def test_cumvar_is_monotone(self):
        """Cumulative variance must be non-decreasing and end at 1.0."""
        from pyannow.step1_svd.procrustes import chopin_cumvar
        events = _make_events(40)
        cv = chopin_cumvar(events, duration_s=10.0)
        assert np.all(np.diff(cv) >= -1e-9), "Cumvar must be non-decreasing"
        assert cv[-1] == pytest.approx(1.0, abs=1e-6), (
            f"Final cumvar must equal 1.0, got {cv[-1]:.6f}")

    def test_k8_cumvar_reported(self):
        """cumvar[7] (k=8) should be a reasonable fraction of total variance."""
        from pyannow.step1_svd.procrustes import chopin_cumvar
        events = _make_events(40, seed=1)
        cv = chopin_cumvar(events, duration_s=10.0)
        if len(cv) >= 8:
            # Just check it's in (0, 1]
            assert 0.0 < cv[7] <= 1.0, f"cumvar@k=8 = {cv[7]:.4f} out of (0,1]"


# ─── procrustes_align ─────────────────────────────────────────────────────────

class TestProcrustesAlign:

    def _make_wc(self, T=100, k_w=4, k_c=4, seed=0):
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((T, k_w)) * np.array([10.0, 1.0, 0.1, 0.01])
        C = rng.standard_normal((T, k_c))
        return W, C

    def test_output_keys(self):
        """Return dict must contain R, residual, singular_values, scale, W_k_scaled."""
        from pyannow.step1_svd.procrustes import procrustes_align
        W, C = self._make_wc()
        r = procrustes_align(W, C)
        for key in ("R", "residual", "singular_values", "scale", "W_k_scaled"):
            assert key in r, f"Missing key '{key}'"

    def test_rotation_is_orthogonal(self):
        """R must satisfy R^T R ≈ I (orthogonality constraint)."""
        from pyannow.step1_svd.procrustes import procrustes_align
        W, C = self._make_wc()
        R = procrustes_align(W, C)["R"]
        RtR = R.T @ R
        np.testing.assert_allclose(
            RtR, np.eye(RtR.shape[0]), atol=1e-10,
            err_msg="R^T R must be identity (R orthogonal)"
        )

    def test_standardize_reduces_residual_on_unscaled_input(self):
        """Standardize=True should give <= residual of standardize=False for ill-scaled W."""
        from pyannow.step1_svd.procrustes import procrustes_align
        W, C = self._make_wc()  # W deliberately ill-scaled (10, 1, 0.1, 0.01)
        r_std  = procrustes_align(W, C, standardize=True)["residual"]
        r_raw  = procrustes_align(W, C, standardize=False)["residual"]
        assert r_std <= r_raw + 1e-9, (
            f"Standardized residual {r_std:.4f} should be ≤ raw {r_raw:.4f} for ill-scaled W")

    def test_scale_vector_shape(self):
        from pyannow.step1_svd.procrustes import procrustes_align
        W, C = self._make_wc(k_w=4)
        scale = procrustes_align(W, C)["scale"]
        assert scale.shape == (4,), f"scale must have shape (k_w,), got {scale.shape}"

    def test_no_standardize_scale_is_ones(self):
        from pyannow.step1_svd.procrustes import procrustes_align
        W, C = self._make_wc()
        scale = procrustes_align(W, C, standardize=False)["scale"]
        np.testing.assert_array_equal(scale, np.ones(W.shape[1]),
                                      "scale must be all 1.0 when standardize=False")
