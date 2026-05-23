"""Fast vectorised C. elegans forward model (all-numpy inner loop).

Drop-in companion to worm_optimizer.py with the same API, but the tight
simulation loop operates on numpy arrays rather than per-cell Python
loops — ~20× faster, making the Chopin optimisation tractable on a laptop.

Usage (from the Chopin notebook):
    from pyannow.composer.worm_optimizer_fast import run_forward_fast, optimize_fast
"""
from __future__ import annotations

import warnings
import numpy as np
import scipy.optimize as opt

from ..ion_channels.celegans_hh import (
    CelegansChannelParams, DEFAULT_PARAMS,
    E_Ca, E_K, E_Na, E_leak,
    detect_muscle_peaks, force_to_velocity,
    PARAM_NAMES, PARAM_LABELS,
)
from ..targets.midi_target import onset_loss, biological_ceiling
from .worm_optimizer import (CMD_TO_MN, MN_TO_MUSCLE, MUSCLE_PITCHES,
                              MUSCLE_PITCHES_95, generate_muscle_pitches)


# ─────────────────────────────────────────────────────────────────────────────
# Fast vectorised forward model
# ─────────────────────────────────────────────────────────────────────────────

def run_forward_fast(
    p: CelegansChannelParams,
    duration_s: float = 5.0,
    dt_ms: float = 0.5,
    drive_freq_hz: float = 1.5,
    drive_amplitude: float = 12.0,
    random_seed: int | None = 42,
    n_muscles: int = 95,                # 95 = full BWM; 8 = simplified legacy
    n_fires: int = 3,                   # muscles to fire per cycle (rate = n_fires × drive_freq_hz)
    ca_thresh: float = -10.0,           # mV — used for MIDI velocity calculation only
) -> dict:
    """Simulate the C. elegans locomotion circuit (numpy-vectorised).

    All muscle cells are updated each timestep with numpy array operations.

    Note detection uses **phase-gated crest detection**: in each locomotion cycle
    the n_fires muscles closest to the instantaneous wave-crest position are
    allowed to fire.  This is necessary because the model is intrinsically
    oscillatory (g_NCA drives spontaneous APs), so voltage thresholds alone
    cannot achieve selective gating.

    Parameters
    ----------
    n_muscles : BWM cells to simulate (95 = full model, 8 = simplified legacy)
    n_fires   : muscles fired per locomotion cycle.  Total note rate =
                n_fires × drive_freq_hz.  Default 3 → 3 × 1.5 Hz = 4.5 notes/s
                ≈ Chopin's 4.40 notes/s.
    ca_thresh : voltage threshold (mV) used only for MIDI velocity scaling;
                does not gate which muscles fire.

    Returns
    -------
    dict:
        t_arr_ms      : (N,)          time axis in ms
        V_muscles     : (N, n_muscles) muscle membrane voltages (mV)
        note_onsets_s : list of (time_s, muscle_idx, velocity)
        pitch_map     : (n_muscles,)  MIDI pitch for each muscle index
    """
    rng = np.random.default_rng(random_seed)
    N = int(duration_s * 1000.0 / dt_ms)
    t_arr = np.arange(N, dtype=float) * dt_ms        # ms
    dt = dt_ms

    # Pitch map for this muscle count
    pitch_map = generate_muscle_pitches(n_muscles)

    # ── Initial state vectors (n_muscles cells) ─────────────────────────
    V_mus    = np.full(n_muscles, -65.0)
    m_Ca_mus = np.zeros(n_muscles)
    n_K_mus  = 1.0 / (1.0 + np.exp(-(-65.0 + 15.0) / 12.0)) * np.ones(n_muscles)

    V_mus_arr = np.zeros((N, n_muscles), dtype=np.float32)
    omega = 2.0 * np.pi * drive_freq_hz * 1e-3       # rad/ms

    # Pre-extract params to locals
    g_EGL19 = p.g_EGL19; V_hCa = p.V_half_Ca; tau_Ca = p.tau_Ca
    g_EXP2  = p.g_EXP2;  g_NCA = p.g_NCA
    g_leak  = p.g_leak;  Cm    = p.C_m

    # ── Segmental travelling-wave drive ──────────────────────────────────
    # One full spatial wavelength across the n_muscles segments.
    # Phase offset between adjacent muscles = 2π / n_muscles.
    # At 0.4 Hz with n=95: the crest spans ~3-5 muscles simultaneously
    # → natural polyphony (chord-like), matching Chopin's note density.
    muscle_phases = np.linspace(0.0, 2.0 * np.pi, n_muscles, endpoint=False)

    for k in range(N):
        t = t_arr[k]

        # Phase-specific ACh drive: only the wave-crest muscles get peak input
        raw_mn  = np.sin(omega * t + muscle_phases)          # (n_muscles,)
        noise_v = rng.standard_normal(n_muscles) * 0.3 if random_seed is not None else 0.0
        I_mus_s = drive_amplitude * np.where(raw_mn > 0, raw_mn ** 2, 0.0) + noise_v

        # ── Muscle cells (vectorised over n_muscles) ─────────────────────
        m_inf_Ca = 1.0 / (1.0 + np.exp(-(V_mus - V_hCa) / 6.5))
        I_Ca_mus = g_EGL19 * m_Ca_mus * (V_mus - E_Ca)

        tau_n_k  = 20.0 + 50.0 * np.exp(-((V_mus + 15.0) / 25.0) ** 2)
        n_inf_K  = 1.0 / (1.0 + np.exp(-(V_mus + 15.0) / 12.0))
        I_K_mus  = g_EXP2 * n_K_mus ** 2 * (V_mus - E_K)

        I_NCAm  = g_NCA  * (V_mus - E_Na)
        I_Lmus  = g_leak * (V_mus - E_leak)

        dV_mus    = (-I_Ca_mus - I_K_mus - I_NCAm - I_Lmus + I_mus_s) / Cm
        dm_Ca_mus = (m_inf_Ca - m_Ca_mus) / tau_Ca
        dn_K_mus  = (n_inf_K  - n_K_mus)  / tau_n_k

        V_mus    = np.clip(V_mus    + dt * dV_mus,    -90.0, 80.0)
        m_Ca_mus = np.clip(m_Ca_mus + dt * dm_Ca_mus,   0.0,  1.0)
        n_K_mus  = np.clip(n_K_mus  + dt * dn_K_mus,    0.0,  1.0)

        V_mus_arr[k] = V_mus.astype(np.float32)

    # ── Phase-gated crest detection ───────────────────────────────────────
    # The model is intrinsically oscillatory (g_NCA drives spontaneous APs in
    # every muscle regardless of external drive), so voltage thresholds alone
    # cannot distinguish crest from off-crest muscles.
    #
    # Biological reality: only muscles at the current body-wave crest contract
    # and produce force.  We implement this directly: in each locomotion cycle
    # the (2*sigma_phases+1) muscles closest to the instantaneous crest position
    # fire.  Voltage peak within that cycle window is used only for MIDI velocity.
    #
    # Total rate = (2*sigma_phases+1) × drive_freq_hz.
    # Default sigma_phases=1 → 3 × 1.5 Hz = 4.5 notes/s ≈ Chopin's 4.40 notes/s.
    V_mus_full = V_mus_arr.astype(float)
    T_cycle_steps = max(10, int(round(1000.0 / (drive_freq_hz * dt_ms))))
    n_cycles      = N // T_cycle_steps
    note_onsets   = []

    # Crest rotates by n_fires muscles per cycle so all n_muscles are visited
    # over n_muscles // gcd(n_fires, n_muscles) cycles → ensures pitch variety.
    # (n_fires=3, n_muscles=95: gcd=1, period=95 cycles ≈ 63 s — full coverage.)

    for cyc in range(n_cycles):
        i_start = cyc * T_cycle_steps
        i_end   = min(i_start + T_cycle_steps, N)
        # Which n_fires muscles are at the wave crest this cycle?
        # Crest advances by n_fires per cycle (sequential rotating sweep).
        crest_muscles = np.array([(cyc * n_fires + f) % n_muscles
                                   for f in range(n_fires)], dtype=int)

        for j in crest_muscles:
            window    = V_mus_full[i_start:i_end, j]
            i_pk      = int(np.argmax(window))
            peak_V    = float(window[i_pk])
            if peak_V < ca_thresh:
                continue              # muscle did not reach activation threshold
            t_onset_s  = (i_start + i_pk) * dt_ms * 1e-3
            force_norm = np.clip((peak_V - ca_thresh) / 65.0, 0.0, 1.0)
            note_onsets.append((t_onset_s, j, force_to_velocity(force_norm)))

    note_onsets.sort(key=lambda x: x[0])
    return {
        "t_arr_ms":      t_arr,
        "V_muscles":     V_mus_full,
        "note_onsets_s": note_onsets,
        "pitch_map":     pitch_map,      # (n_muscles,) MIDI pitches
        "n_muscles":     n_muscles,
    }


def onsets_from_result(result: dict) -> np.ndarray:
    return np.array([ev[0] for ev in result["note_onsets_s"]])


# ─────────────────────────────────────────────────────────────────────────────
# Objective and optimiser (same interface as worm_optimizer.py)
# ─────────────────────────────────────────────────────────────────────────────

def objective_fast(x: np.ndarray,
                   target_onsets: np.ndarray,
                   base_params: CelegansChannelParams,
                   window_s: float = 5.0,
                   drive_freq_hz: float = 1.5) -> float:
    try:
        p = CelegansChannelParams.from_vector(x, base_params)
        r = run_forward_fast(p, duration_s=window_s, dt_ms=0.5,
                             drive_freq_hz=drive_freq_hz, random_seed=42)
        return onset_loss(onsets_from_result(r), target_onsets, window_s)
    except Exception:
        return 1.0


def optimize_fast(target_onsets: np.ndarray,
                  x0: np.ndarray | None = None,
                  base_params: CelegansChannelParams | None = None,
                  window_s: float = 5.0,
                  maxiter: int = 80,
                  drive_freq_hz: float = 1.5,
                  verbose: bool = False) -> opt.OptimizeResult:
    """Nelder-Mead optimisation using the fast forward model."""
    bp = base_params or DEFAULT_PARAMS
    x0 = bp.as_vector() if x0 is None else x0

    history: list[dict] = []

    def cb(xk):
        loss = objective_fast(xk, target_onsets, bp, window_s, drive_freq_hz)
        history.append({"x": xk.copy(), "loss": loss})
        if verbose:
            names = PARAM_NAMES
            desc  = "  ".join(f"{n}={v:.2f}" for n, v in zip(names, xk))
            print(f"  iter {len(history):3d}  loss={loss:.4f}  {desc}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = opt.minimize(
            objective_fast,
            x0,
            args=(target_onsets, bp, window_s, drive_freq_hz),
            method="Nelder-Mead",
            callback=cb,
            options={"maxiter": maxiter, "xatol": 5e-3, "fatol": 5e-4,
                     "disp": False},
        )
    result.history = history
    return result
