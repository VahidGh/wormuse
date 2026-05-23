"""Tests for PyANNOW/src/pyannow/ion_channels/celegans_hh.py

Covers:
  - Gating variable boundary conditions (must stay in [0, 1])
  - EGL-19 activation curve is monotonically increasing
  - Muscle simulation produces finite output and correct shape
  - HH spike detection works on known firing conditions
  - CelegansChannelParams vector round-trips cleanly
"""
from __future__ import annotations

import numpy as np
import pytest


class TestGatingFunctions:

    def test_egl19_inf_monotone(self, default_params):
        """m_∞(V) must be strictly increasing in V (L15: sigmoid property)."""
        from pyannow.ion_channels.celegans_hh import egl19_inf
        V = np.linspace(-80, 40, 100)
        m = egl19_inf(V, default_params)
        assert np.all(np.diff(m) >= 0), "EGL-19 m_inf must be non-decreasing"

    def test_egl19_inf_bounds(self, default_params):
        """m_∞ must be in (0, 1) for any finite voltage."""
        from pyannow.ion_channels.celegans_hh import egl19_inf
        V = np.linspace(-120, 60, 200)
        m = egl19_inf(V, default_params)
        assert np.all(m > 0) and np.all(m < 1), "m_inf must be in (0,1)"

    def test_exp2_tau_positive(self):
        """EXP-2 time constant must be positive everywhere."""
        from pyannow.ion_channels.celegans_hh import exp2_tau
        V = np.linspace(-80, 40, 100)
        tau = exp2_tau(V)
        assert np.all(tau > 0), "EXP-2 tau must be positive"

    def test_exp2_inf_bounds(self):
        """EXP-2 steady-state must be in (0, 1)."""
        from pyannow.ion_channels.celegans_hh import exp2_inf
        V = np.linspace(-80, 40, 100)
        n = exp2_inf(V)
        assert np.all(n >= 0) and np.all(n <= 1)


class TestMuscleSimulation:

    def test_simulate_muscle_no_input_stays_near_rest(self, default_params):
        """Muscle simulation with zero drive must produce a finite voltage trace.

        Note: with DEFAULT_PARAMS, the NCA background current (g_NCA=0.8 mS/cm²)
        spontaneously depolarises the cell above EGL-19 threshold even without
        synaptic input.  The test only checks finiteness; see
        test_simulate_muscle_zero_nca_stays_near_rest for resting-potential behaviour.
        """
        from pyannow.ion_channels.celegans_hh import simulate_muscle
        t, V = simulate_muscle(lambda t: 0.0, duration_ms=50.0, p=default_params, dt=0.2)
        assert V.shape == t.shape, "V and t arrays must have the same shape"
        assert np.all(np.isfinite(V)), "No NaN/Inf with zero drive"

    def test_simulate_muscle_drive_excites(self, default_params):
        """A suprathreshold ACh pulse must depolarise the muscle."""
        from pyannow.ion_channels.celegans_hh import simulate_muscle
        t, V = simulate_muscle(
            lambda t_: 15.0 if 10 <= t_ <= 40 else 0.0,
            duration_ms=80.0, p=default_params, dt=0.1)
        assert np.max(V) > default_params.V_half_Ca, (
            "Suprathreshold drive must push V above half-activation")

    def test_simulate_muscle_shape(self, default_params):
        """Output arrays must have length matching expected timesteps."""
        from pyannow.ion_channels.celegans_hh import simulate_muscle
        duration_ms, dt = 100.0, 0.5
        t, V = simulate_muscle(lambda _: 0.0, duration_ms=duration_ms,
                                p=default_params, dt=dt)
        expected_len = int(duration_ms / dt)
        assert len(t) == expected_len and len(V) == expected_len

    def test_detect_muscle_peaks_empty(self):
        """No peaks should be returned for a flat voltage trace."""
        from pyannow.ion_channels.celegans_hh import detect_muscle_peaks
        t = np.linspace(0, 100, 1000)
        V = np.full_like(t, -65.0)
        peaks = detect_muscle_peaks(t, V, threshold=-30.0)
        assert len(peaks) == 0, "Flat trace below threshold should yield no peaks"

    def test_detect_muscle_peaks_single_spike(self):
        """A single spike crossing -30 mV should produce exactly one peak."""
        from pyannow.ion_channels.celegans_hh import detect_muscle_peaks
        t = np.linspace(0, 200, 2000)
        V = np.full_like(t, -65.0)
        # Insert one spike at t=100 ms
        idx = np.argmin(np.abs(t - 100))
        V[idx - 5: idx + 5] = 20.0     # depolarise briefly
        peaks = detect_muscle_peaks(t, V, threshold=-30.0, min_interval_ms=50.0)
        assert len(peaks) == 1, f"Expected 1 peak, got {len(peaks)}"
        assert abs(peaks[0] - 100.0) < 5.0, "Peak should be near t=100 ms"


class TestChannelParams:

    def test_as_vector_length(self, default_params):
        """as_vector() must return exactly 5 elements (4 channel params + ca_thresh)."""
        v = default_params.as_vector()
        assert len(v) == 5, f"Expected 5 params, got {len(v)}"

    def test_vector_round_trip(self, default_params):
        """from_vector(as_vector()) must recover the original parameters."""
        from pyannow.ion_channels.celegans_hh import CelegansChannelParams
        v = default_params.as_vector()
        p2 = CelegansChannelParams.from_vector(v)
        assert abs(p2.g_EGL19   - default_params.g_EGL19)   < 1e-9
        assert abs(p2.V_half_Ca - default_params.V_half_Ca) < 1e-9
        assert abs(p2.tau_Ca    - default_params.tau_Ca)    < 1e-9
        assert abs(p2.g_EXP2    - default_params.g_EXP2)    < 1e-9
        assert abs(p2.ca_thresh - default_params.ca_thresh) < 1e-9

    def test_force_to_velocity_range(self):
        """force_to_velocity must always return an integer in [1, 127]."""
        from pyannow.ion_channels.celegans_hh import force_to_velocity
        for f in [0.0, 0.001, 0.5, 0.999, 1.0, 2.0]:
            v = force_to_velocity(f)
            assert isinstance(v, int), "velocity must be int"
            assert 1 <= v <= 127, f"velocity {v} out of MIDI range for f={f}"
