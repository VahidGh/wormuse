"""Tests for pyannow.training.cv — time-series cross-validation.

Covers (ISSUE-038, v0.8.0):
  - time_series_cv() shape, reproducibility, non-leakage
  - blocked_bootstrap_ci() CI validity
"""
from __future__ import annotations

import numpy as np
import pytest


# ─── Minimal fit/score stubs ──────────────────────────────────────────────────

class _IdentityModel:
    """Trivial model: predicts Z itself (no actual learning)."""
    def __init__(self, coef):
        self.coef = coef

    def predict(self, Z):
        return Z @ self.coef


def _fit_fn(Z, C):
    """Least-squares fit: coef = pinv(Z) @ C."""
    coef, *_ = np.linalg.lstsq(Z, C, rcond=None)
    return _IdentityModel(coef)


def _score_fn(model, Z, C):
    """Return negative MSE (higher = better)."""
    C_hat = model.predict(Z)
    return -float(np.mean((C - C_hat) ** 2))


def _make_data(T=100, d_Z=4, d_C=4, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((T, d_Z))
    C = Z @ rng.standard_normal((d_Z, d_C)) + rng.standard_normal((T, d_C)) * 0.1
    return Z, C


# ─── time_series_cv ───────────────────────────────────────────────────────────

class TestTimeSeriesCV:

    def test_returns_expected_keys(self):
        from pyannow.training.cv import time_series_cv
        Z, C = _make_data()
        r = time_series_cv(_fit_fn, _score_fn, Z, C, n_splits=5)
        for key in ("mean", "std", "scores", "n_folds", "fold_info"):
            assert key in r, f"Missing key '{key}'"

    def test_n_folds_leq_n_splits(self):
        """Number of evaluated folds must be ≤ n_splits (some may be skipped)."""
        from pyannow.training.cv import time_series_cv
        Z, C = _make_data(T=100)
        r = time_series_cv(_fit_fn, _score_fn, Z, C, n_splits=5)
        assert 1 <= r["n_folds"] <= 5, f"n_folds={r['n_folds']} out of expected range"

    def test_scores_list_matches_n_folds(self):
        from pyannow.training.cv import time_series_cv
        Z, C = _make_data()
        r = time_series_cv(_fit_fn, _score_fn, Z, C, n_splits=4)
        assert len(r["scores"]) == r["n_folds"]
        assert len(r["fold_info"]) == r["n_folds"]

    def test_no_future_leakage(self):
        """Each test fold's start must be strictly after all train indices."""
        from pyannow.training.cv import time_series_cv
        Z, C = _make_data(T=200)
        r = time_series_cv(_fit_fn, _score_fn, Z, C, n_splits=5)
        for fi in r["fold_info"]:
            assert fi["train_size"] > 0, "Train set must not be empty"
            # train indices are [0, test_start), test indices are [test_start, test_end)
            assert fi["test_start"] > 0, (
                f"test_start={fi['test_start']} must be > 0 (train must precede test)")

    def test_deterministic(self):
        """Same inputs must give identical results every run."""
        from pyannow.training.cv import time_series_cv
        Z, C = _make_data()
        r1 = time_series_cv(_fit_fn, _score_fn, Z, C, n_splits=5)
        r2 = time_series_cv(_fit_fn, _score_fn, Z, C, n_splits=5)
        assert r1["mean"]   == pytest.approx(r2["mean"],  abs=1e-12)
        assert r1["scores"] == pytest.approx(r2["scores"], abs=1e-12)

    def test_raises_on_too_short_data(self):
        """T < n_splits * 2 must raise ValueError."""
        from pyannow.training.cv import time_series_cv
        Z, C = _make_data(T=5)
        with pytest.raises(ValueError, match="too short"):
            time_series_cv(_fit_fn, _score_fn, Z, C, n_splits=5)

    def test_std_is_nonnegative(self):
        from pyannow.training.cv import time_series_cv
        Z, C = _make_data()
        r = time_series_cv(_fit_fn, _score_fn, Z, C, n_splits=5)
        assert r["std"] >= 0.0


# ─── blocked_bootstrap_ci ─────────────────────────────────────────────────────

class TestBlockedBootstrapCI:

    def test_returns_expected_keys(self):
        from pyannow.training.cv import blocked_bootstrap_ci
        Z, C = _make_data(T=100)
        r = blocked_bootstrap_ci(_fit_fn, _score_fn, Z, C, block_len=10, n_boot=50)
        for key in ("mean", "ci_low", "ci_high", "std", "n_boot"):
            assert key in r, f"Missing key '{key}'"

    def test_ci_contains_mean(self):
        from pyannow.training.cv import blocked_bootstrap_ci
        Z, C = _make_data(T=100)
        r = blocked_bootstrap_ci(_fit_fn, _score_fn, Z, C, block_len=10, n_boot=100)
        assert r["ci_low"] <= r["mean"] <= r["ci_high"], (
            f"CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}] does not contain mean {r['mean']:.4f}")

    def test_n_boot_stored_correctly(self):
        from pyannow.training.cv import blocked_bootstrap_ci
        Z, C = _make_data(T=100)
        r = blocked_bootstrap_ci(_fit_fn, _score_fn, Z, C, block_len=10, n_boot=77)
        assert r["n_boot"] == 77

    def test_deterministic_with_seed(self):
        from pyannow.training.cv import blocked_bootstrap_ci
        Z, C = _make_data(T=120)
        r1 = blocked_bootstrap_ci(_fit_fn, _score_fn, Z, C, block_len=10, n_boot=50, seed=7)
        r2 = blocked_bootstrap_ci(_fit_fn, _score_fn, Z, C, block_len=10, n_boot=50, seed=7)
        assert r1["mean"] == pytest.approx(r2["mean"], abs=1e-12)

    def test_raises_on_too_short_for_blocks(self):
        from pyannow.training.cv import blocked_bootstrap_ci
        Z, C = _make_data(T=10)
        with pytest.raises(ValueError):
            blocked_bootstrap_ci(_fit_fn, _score_fn, Z, C, block_len=10, n_boot=10)
