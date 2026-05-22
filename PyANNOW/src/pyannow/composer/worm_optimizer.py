"""Worm-to-Chopin forward model and optimiser.

The pipeline:
  1. Command interneurons receive a rhythmic drive (I_cmd).
  2. Motor neurons (B-class for forward, A-class for backward) receive
     graded input from command interneurons via the connectome.
  3. Each motor neuron drives a muscle group via ACh release.
  4. The muscle cell (EGL-19 + EXP-2) integrates synaptic input and produces
     a Ca²⁺ action potential — one spike = one piano 'note onset'.
  5. The note's pitch = muscle-group index mapped to a scale.
  6. The note's velocity = peak Ca²⁺ current amplitude (force), rescaled
     to MIDI range 0-127.

C. elegans locomotion circuit (simplified)
------------------------------------------
The full 302-neuron connectome is reduced to 12 key neurons from the
locomotion circuit (White 1986; Chalfie 1985; Kato 2015):

  Command interneurons (4):
    AVA / AVD  — backward command (active during reversals)
    AVB / PVC  — forward command (active during forward crawling)

  A-class motor neurons (4):  VA1, VA2, DA1, DA2
    → drive ventral/dorsal BWMs for backward motion
  B-class motor neurons (4):  VB1, VB2, DB1, DB2
    → drive ventral/dorsal BWMs for forward motion

  Body wall muscle groups (8):
    D1, D2 (dorsal anterior), D3, D4 (dorsal posterior)
    V1, V2 (ventral anterior), V3, V4 (ventral posterior)
    → each assigned a note on a pentatonic scale (rescalable)

Ion channels that matter most (PINN-tunable)
--------------------------------------------
  g_EGL19   — maximal Ca²⁺ conductance → force/loudness
  V_half_Ca — EGL-19 activation threshold → firing sensitivity
  tau_Ca    — Ca²⁺ activation speed → note attack / tempo ceiling
  g_EXP2    — K⁺ repolarisation → note duration / release

References
----------
White et al. 1986; Kato et al. 2015; Boyle et al. 2012; Jospin et al. 2002.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.optimize as opt

from ..ion_channels.celegans_hh import (
    CelegansChannelParams, DEFAULT_PARAMS,
    muscle_rhs, resting_state, detect_muscle_peaks,
    motorneuron_rhs, motorneuron_ntransmitter,
    PARAM_NAMES, PARAM_LABELS,
)
from ..targets.midi_target import onset_loss, note_rate_mismatch, biological_ceiling

# ─────────────────────────────────────────────────────────────────────────────
# Connectome weights  (subset — sign-correct, magnitude fitted to Boyle 2012)
# ─────────────────────────────────────────────────────────────────────────────

# cmd_to_motorneuron[i][j] = synaptic weight from command IN i to motor neuron j
# Rows: [AVA, AVD, AVB, PVC]
# Cols: [VA1, VA2, DA1, DA2, VB1, VB2, DB1, DB2]
CMD_TO_MN = np.array([
    #   VA1  VA2  DA1  DA2  VB1  VB2  DB1  DB2
    [  5.0,  4.0,  3.0,  3.0,  0.0,  0.0,  0.0,  0.0],  # AVA → A-class (backward)
    [  3.0,  3.0,  2.0,  2.0,  0.0,  0.0,  0.0,  0.0],  # AVD → A-class (backward)
    [  0.0,  0.0,  0.0,  0.0,  5.0,  4.0,  3.0,  3.0],  # AVB → B-class (forward)
    [  0.0,  0.0,  0.0,  0.0,  3.0,  3.0,  2.0,  2.0],  # PVC → B-class (forward)
])  # units: pS / nS normalised; will be scaled by synaptic input

# Motor neuron → muscle mapping  (which MN excites which muscle group)
# Rows: motor neurons [VA1, VA2, DA1, DA2, VB1, VB2, DB1, DB2]
# Cols: muscle groups [D1, D2, D3, D4, V1, V2, V3, V4]
MN_TO_MUSCLE = np.array([
    #  D1   D2   D3   D4   V1   V2   V3   V4
    [  0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0],  # VA1 → V anterior
    [  0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 1.0, 0.0],  # VA2 → V ant-mid
    [  4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # DA1 → D anterior
    [  2.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # DA2 → D ant-mid
    [  0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 2.0],  # VB1 → V posterior
    [  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0],  # VB2 → V post
    [  0.0, 2.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0],  # DB1 → D posterior
    [  0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0],  # DB2 → D post
])


# ─────────────────────────────────────────────────────────────────────────────
# Note-pitch assignment  (muscle group → MIDI pitch)
# Force amplitude is rescaled to MIDI velocity (0-127).
# ─────────────────────────────────────────────────────────────────────────────

def generate_muscle_pitches(n_muscles: int = 95,
                             key: str = "C#m",
                             midi_lo: int = 25,
                             midi_hi: int = 108) -> np.ndarray:
    """Return an array of `n_muscles` MIDI pitches distributed across the piano.

    Biological layout of the 95 C. elegans BWM cells:
      ~24 dorsal-left   (head → tail)  → lowest pitches  (bass register)
      ~23 dorsal-right  (head → tail)  → lower-mid
      ~24 ventral-left  (head → tail)  → upper-mid
      ~24 ventral-right (head → tail)  → highest pitches (treble register)

    The wave of contraction propagates head-to-tail, so as it passes through
    each quadrant the pitch rises from bass to treble — a natural physical
    interpretation.

    With n_muscles=8  → the original C# minor pentatonic (simplified model)
    With n_muscles=95 → the full 95-cell map across MIDI 25-108

    Parameters
    ----------
    n_muscles : number of muscle groups (8 for simplified; 95 for full model)
    key       : "C#m" = C# natural minor scale; "chrom" = chromatic
    midi_lo   : lowest MIDI pitch to use (default 25 = C#1, lowest Nocturne bass note)
    midi_hi   : highest MIDI pitch to use (default 108 = C7, upper piano range)
    """
    if n_muscles == 8 and key == "C#m":
        # Original simplified mapping — preserved for backward compatibility
        return np.array([61, 64, 66, 68, 71, 73, 76, 78])

    if key == "C#m":
        # C# natural minor scale intervals from C# (semitones from root):
        # C# D# E F# G# A B  =  0 2 3 5 7 8 10
        intervals = [0, 2, 3, 5, 7, 8, 10]
        scale_pitches = []
        for octave in range(11):         # 0-10 covers the full MIDI range
            for semi in intervals:
                p = octave * 12 + 1 + semi   # C# = 1 (C=0)
                if midi_lo <= p <= midi_hi:
                    scale_pitches.append(p)
        pool = np.array(sorted(scale_pitches))
    else:
        pool = np.arange(midi_lo, midi_hi + 1)

    # Distribute n_muscles evenly across the pool (always stays within piano range)
    if n_muscles <= len(pool):
        idx = np.round(np.linspace(0, len(pool) - 1, n_muscles)).astype(int)
        return pool[idx]
    else:
        # More muscles than scale notes — fill with chromatic notes, staying ≤ midi_hi
        chrom = np.arange(midi_lo, midi_hi + 1)
        idx = np.round(np.linspace(0, len(chrom) - 1, n_muscles)).astype(int)
        return chrom[idx]


# Default pitch arrays
MUSCLE_PITCHES    = generate_muscle_pitches(n_muscles=8)   # 8-cell simplified (backward compat)
MUSCLE_PITCHES_95 = generate_muscle_pitches(n_muscles=95)  # 95-cell full BWM


def force_to_velocity(force_normalised: float,
                      force_scale: float = 50.0) -> int:
    """Convert normalised muscle force to MIDI velocity 1-127.

    force_scale is the amplification factor that makes the worm's tiny forces
    comparable to a human pianist. This is the 'size rescaling' mentioned in
    the project spec.
    """
    v = int(np.clip(force_normalised * force_scale, 1, 127))
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Forward model
# ─────────────────────────────────────────────────────────────────────────────

def run_forward(p: CelegansChannelParams,
                duration_ms: float = 15000.0,
                dt: float = 0.1,
                drive_freq_hz: float = 1.5,
                drive_amplitude: float = 12.0,
                random_seed: int | None = None) -> dict:
    """Simulate the full worm locomotion circuit for `duration_ms` ms.

    Parameters
    ----------
    p               : ion channel parameters (the PINN-tunable variables)
    duration_ms     : simulation duration
    dt              : timestep in ms (≤0.2 ms recommended for stability)
    drive_freq_hz   : frequency of the rhythmic command-interneuron drive (Hz)
    drive_amplitude : amplitude of I_cmd (μA/cm²) — set by the circuit drive
    random_seed     : if set, add small Gaussian noise to break symmetry

    Returns
    -------
    dict with keys:
        t_arr         : (N,) time array, ms
        V_muscles     : (N, 8) muscle voltages
        note_onsets_s : list of (time_s, muscle_idx, velocity) tuples
    """
    rng = np.random.default_rng(random_seed)
    t_arr = np.arange(0, duration_ms, dt)
    N = len(t_arr)

    # State: [V_cmd×4, V_mn×8, V_mus×8, m_Ca_mn×8, m_Ca_mus×8, n_K_mus×8]
    #  Indices: cmd 0..3, mn 4..11, mus 12..19, m_Ca_mn 20..27,
    #           m_Ca_mus 28..35, n_K_mus 36..43
    state = np.zeros(44)
    state[0:4]   = -65.0                              # cmd interneuron resting V
    state[4:12]  = -65.0                              # motor neuron resting V
    state[12:20] = resting_state(p)[0]                # muscle V
    state[20:28] = 1.0 / (1.0 + np.exp(-(-65 + 30) / 5.0))   # m_Ca_mn
    state[28:36] = p.V_half_Ca / (-p.V_half_Ca + 1e-3)        # m_Ca_mus (EGL-19)
    state[28:36] = np.clip(
        1.0 / (1.0 + np.exp(-(resting_state(p)[0] - p.V_half_Ca) / 6.5)),
        0, 1)
    state[36:44] = 1.0 / (1.0 + np.exp(-(-65 + 15) / 12.0))  # n_K_mus

    V_muscles_arr = np.zeros((N, 8))
    omega = 2.0 * np.pi * drive_freq_hz * 1e-3   # rad/ms

    # Phase offsets for 4 command interneurons (90° apart → body wave)
    cmd_phases = np.array([0.0, np.pi * 0.5, np.pi, np.pi * 1.5])

    for k, t in enumerate(t_arr):
        # ─ Command interneurons: sinusoidal rhythmic drive ────────────────
        I_cmd_vec = drive_amplitude * np.sin(omega * t + cmd_phases)
        if random_seed is not None:
            I_cmd_vec += rng.normal(0, 0.5, 4)

        # Update cmd interneurons (very simplified — just track instantaneous V)
        V_cmd = np.tanh(I_cmd_vec / 10.0) * 20.0 - 45.0   # graded output

        # ─ Motor neurons: graded input from commands via connectome ───────
        I_mn = CMD_TO_MN.T @ (V_cmd - (-65.0))    # (8,) synaptic current
        for j in range(8):
            mn_state = state[4 + j:5 + j + 1]      # V_mn, m_Ca_mn
            mn_state_2 = np.array([state[4 + j], state[20 + j]])
            rhs = motorneuron_rhs(t, mn_state_2, I_mn[j], p)
            state[4 + j]  += dt * rhs[0]
            state[20 + j] += dt * rhs[1]
        state[4:12] = np.clip(state[4:12], -90, 60)
        state[20:28] = np.clip(state[20:28], 0, 1)

        # ─ Neurotransmitter release and muscle drive ──────────────────────
        V_mn = state[4:12]
        nt_release = np.array([motorneuron_ntransmitter(v) for v in V_mn])
        I_muscle = MN_TO_MUSCLE.T @ nt_release   # (8,) ACh drive to each muscle

        # ─ Muscle cells: EGL-19 + EXP-2 dynamics ─────────────────────────
        for j in range(8):
            mus_state = np.array([state[12 + j], state[28 + j], state[36 + j]])
            I_s = float(I_muscle[j]) * 3.0          # ACh → current (scaled)
            rhs = muscle_rhs(t, mus_state, I_s, p)
            state[12 + j] += dt * rhs[0]
            state[28 + j] += dt * rhs[1]
            state[36 + j] += dt * rhs[2]
        state[12:20] = np.clip(state[12:20], -90, 80)
        state[28:36] = np.clip(state[28:36], 0, 1)
        state[36:44] = np.clip(state[36:44], 0, 1)

        V_muscles_arr[k, :] = state[12:20]

    # ─ Detect note events ─────────────────────────────────────────────────
    note_onsets = []
    for j in range(8):
        peaks_ms = detect_muscle_peaks(t_arr, V_muscles_arr[:, j])
        for pk_ms in peaks_ms:
            # Force proxy: max V during the spike (normalised to [0,1])
            i_pk = int(pk_ms / dt)
            win = slice(max(0, i_pk - 20), min(N, i_pk + 30))
            peak_V = float(V_muscles_arr[win, j].max())
            force_norm = np.clip((peak_V + 30.0) / 80.0, 0, 1)
            vel = force_to_velocity(force_norm)
            note_onsets.append((float(pk_ms) * 1e-3, j, vel))

    note_onsets.sort(key=lambda x: x[0])
    return {
        "t_arr":          t_arr,
        "V_muscles":      V_muscles_arr,
        "note_onsets_s":  note_onsets,
    }


def onsets_from_result(result: dict) -> np.ndarray:
    """Extract a sorted 1-D array of onset times (seconds) from run_forward output."""
    return np.array([ev[0] for ev in result["note_onsets_s"]])


# ─────────────────────────────────────────────────────────────────────────────
# Objective function and Nelder-Mead optimiser
# ─────────────────────────────────────────────────────────────────────────────

def objective(x: np.ndarray,
              target_onsets: np.ndarray,
              base_params: CelegansChannelParams,
              window_s: float = 15.0,
              drive_freq_hz: float = 1.5) -> float:
    """Loss: onset_loss between worm output and Chopin target."""
    try:
        p = CelegansChannelParams.from_vector(x, base_params)
        result = run_forward(p, duration_ms=window_s * 1000.0,
                             dt=0.1, drive_freq_hz=drive_freq_hz,
                             random_seed=42)
        worm_onsets = onsets_from_result(result)
        return onset_loss(worm_onsets, target_onsets, window_s)
    except Exception:
        return 1.0    # penalty for bad params


def optimize_to_chopin(target_onsets: np.ndarray,
                       x0: np.ndarray | None = None,
                       base_params: CelegansChannelParams | None = None,
                       window_s: float = 15.0,
                       maxiter: int = 120,
                       drive_freq_hz: float = 1.5,
                       callback=None) -> opt.OptimizeResult:
    """Nelder-Mead optimisation of ion-channel parameters toward the Chopin target.

    Parameters
    ----------
    target_onsets : sorted array of Chopin note onset times (seconds)
    x0            : initial parameter vector [g_EGL19, V_half_Ca, tau_Ca, g_EXP2]
                    Default: DEFAULT_PARAMS.as_vector()
    window_s      : length of the simulation window (shorter = faster iteration)
    maxiter       : max Nelder-Mead iterations
    callback      : optional f(xk) called after each iteration
    """
    bp = base_params or DEFAULT_PARAMS
    x0 = x0 if x0 is not None else bp.as_vector()

    history = []

    def cb(xk):
        loss = objective(xk, target_onsets, bp, window_s, drive_freq_hz)
        history.append({"x": xk.copy(), "loss": loss})
        if callback:
            callback(xk, loss, len(history))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = opt.minimize(
            objective,
            x0,
            args=(target_onsets, bp, window_s, drive_freq_hz),
            method="Nelder-Mead",
            callback=cb,
            options={
                "maxiter": maxiter,
                "xatol": 1e-3,
                "fatol": 1e-4,
                "disp": False,
            },
        )

    result.history = history
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity analysis  (compute d(loss)/d(param_i) by finite difference)
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_analysis(x_opt: np.ndarray,
                         target_onsets: np.ndarray,
                         base_params: CelegansChannelParams,
                         window_s: float = 15.0,
                         eps_frac: float = 0.10) -> dict:
    """Finite-difference sensitivity: how much does each ion-channel parameter
    move the loss when perturbed by ±10%?

    Returns a dict with 'sensitivities' (relative) and 'labels'.
    """
    L0 = objective(x_opt, target_onsets, base_params, window_s)
    sens = np.zeros(len(x_opt))
    for i in range(len(x_opt)):
        dx = x_opt.copy()
        dx[i] *= (1.0 + eps_frac)
        L_plus  = objective(dx, target_onsets, base_params, window_s)
        dx[i] = x_opt[i] * (1.0 - eps_frac)
        L_minus = objective(dx, target_onsets, base_params, window_s)
        sens[i] = abs(L_plus - L_minus) / (2.0 * eps_frac * abs(x_opt[i]) + 1e-9)

    # Normalise to [0, 1]
    if sens.max() > 0:
        sens_norm = sens / sens.max()
    else:
        sens_norm = sens

    return {
        "raw":          sens,
        "normalised":   sens_norm,
        "labels":       PARAM_LABELS,
        "param_names":  PARAM_NAMES,
        "L0":           L0,
    }
