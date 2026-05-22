"""Shared pytest fixtures for PyANNOW tests.

All fixtures are fast (no real simulation) and self-contained.
The optional MIDI_PATH fixture skips if the asset file is absent so
tests pass even in CI environments without the full repo data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ── Repo root (relative to this file) ─────────────────────────────────────
REPO_ROOT = Path(__file__).parents[2]   # PyANNOW/tests/ → PyANNOW/ → wormuse/
MIDI_FILE = REPO_ROOT / "shared/examples/chopin_nocturne_op_posth_csharp_minor.mid"


# ── Channel parameters ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def default_params():
    from pyannow.ion_channels.celegans_hh import DEFAULT_PARAMS
    return DEFAULT_PARAMS


@pytest.fixture(scope="session")
def fast_params():
    """Parameters with small τ_Ca (5 ms) for quicker muscle tests."""
    from pyannow.ion_channels.celegans_hh import CelegansChannelParams
    return CelegansChannelParams(tau_Ca=5.0, g_EGL19=8.0)


# ── Synthetic neural activity ──────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_X_neural():
    """302 × 500 synthetic neural activity matrix (deterministic)."""
    rng = np.random.default_rng(42)
    T = 500
    X = np.zeros((302, T))
    for i in range(302):
        X[i] = rng.standard_normal(T) * 0.1 + np.sin(
            np.linspace(0, 4 * np.pi, T) + i * 0.1)
    return X


@pytest.fixture(scope="session")
def synthetic_t_arr_ms():
    """Time axis in ms for the synthetic neural activity (T=500, dt=0.5ms)."""
    return np.arange(500, dtype=float) * 0.5


# ── Tiny synthetic MIDI-like events ───────────────────────────────────────

@pytest.fixture
def tiny_events():
    """5 synthetic NoteEvent-like objects for testing synthesise_melody."""
    from dataclasses import dataclass

    @dataclass
    class FakeEvent:
        time_s: float
        pitch: int
        velocity: int
        duration: float

    return [
        FakeEvent(0.0,  61, 70, 0.5),
        FakeEvent(0.5,  64, 65, 0.5),
        FakeEvent(1.0,  66, 72, 0.5),
        FakeEvent(1.5,  68, 68, 0.5),
        FakeEvent(2.0,  71, 60, 0.5),
    ]


# ── Real MIDI path (skips if file not present) ────────────────────────────

@pytest.fixture(scope="session")
def midi_path():
    if not MIDI_FILE.exists():
        pytest.skip(f"MIDI file not found: {MIDI_FILE}")
    return MIDI_FILE
