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


# ─────────────────────────────────────────────────────────────────────────────
# Boyle et al. (2012) 4×24 muscle quadrant layout
# ─────────────────────────────────────────────────────────────────────────────

BOYLE_QUADRANT_LAYOUT = {
    "reference": "Boyle et al. 2012, 'Gait Modulation in C. elegans: An Integrated Neuromechanical Model'",
    "n_quadrants": 4,
    "n_per_quadrant": 24,
    "n_total": 96,
    "quadrants": {
        "DL": {"label": "Dorsal-Left",   "indices": (0,  24), "midi_range": (24,  47), "octaves": "C1-B2"},
        "VL": {"label": "Ventral-Left",  "indices": (24, 48), "midi_range": (48,  71), "octaves": "C3-B4"},
        "DR": {"label": "Dorsal-Right",  "indices": (48, 72), "midi_range": (72,  95), "octaves": "C5-B6"},
        "VR": {"label": "Ventral-Right", "indices": (72, 96), "midi_range": (96, 119), "octaves": "C7-B8"},
    },
    "musical_mapping": (
        "4 quadrants × 24 muscles = 96 muscle cells ≡ 8 octaves × 12 semitones = 96 piano keys. "
        "Head-to-tail wave propagation → pitch rises bass→treble within each quadrant. "
        "DL/DR (dorsal) fire in phase; VL/VR (ventral) fire 180° out of phase (antiphase body wave). "
        "This maps directly to piano keyboard structure: each quadrant occupies 2 chromatic octaves."
    ),
}


def generate_muscle_pitches(n_muscles: int = 96,
                             key: str = "C#m",
                             midi_lo: int = 25,
                             midi_hi: int = 108) -> np.ndarray:
    """Return an array of `n_muscles` MIDI pitches distributed across the piano.

    Biological layout per Boyle et al. (2012):
      n_muscles=96 → 4 quadrants × 24 cells, chromatic 8-octave mapping:
        DL (0-23)  : C1-B2  (MIDI 24-47)  — bass register
        VL (24-47) : C3-B4  (MIDI 48-71)  — lower-mid
        DR (48-71) : C5-B6  (MIDI 72-95)  — upper-mid
        VR (72-95) : C7-B8  (MIDI 96-119) — treble register
      96 cells = 8 octaves × 12 semitones — maps directly to the piano keyboard.

      n_muscles=95 → original 2-quadrant model, C#m scale across MIDI 25-108
      n_muscles=8  → original 8-cell C# minor pentatonic (backward compat)

    Parameters
    ----------
    n_muscles : number of muscle groups (8 / 95 for legacy; 96 for Boyle architecture)
    key       : "C#m" = C# natural minor scale; "chrom" = chromatic; ignored for n=96
    midi_lo   : lowest MIDI pitch (ignored for n=96, which uses quadrant-based ranges)
    midi_hi   : highest MIDI pitch (ignored for n=96)
    """
    if n_muscles == 8 and key == "C#m":
        # Original simplified mapping — preserved for backward compatibility
        return np.array([61, 64, 66, 68, 71, 73, 76, 78])

    if n_muscles == 96:
        # Boyle et al. 4×24 quadrant layout — chromatic 8-octave mapping
        # Each quadrant gets 2 chromatic octaves = 24 consecutive semitones.
        # DL→C1-B2 (24-47), VL→C3-B4 (48-71), DR→C5-B6 (72-95), VR→C7-B8 (96-119)
        return np.concatenate([
            np.arange(24, 48),   # DL: C1-B2
            np.arange(48, 72),   # VL: C3-B4
            np.arange(72, 96),   # DR: C5-B6
            np.arange(96, 120),  # VR: C7-B8
        ]).astype(int)

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
MUSCLE_PITCHES_95 = generate_muscle_pitches(n_muscles=95)  # 95-cell full BWM (2-quadrant)
MUSCLE_PITCHES_96 = generate_muscle_pitches(n_muscles=96)  # 96-cell Boyle 4×24 (v0.7.0)


# ─────────────────────────────────────────────────────────────────────────────
# 302-neuron biologically structured synthetic activity  (v0.7.0)
# ─────────────────────────────────────────────────────────────────────────────

def generate_neural_activity_302(
    n_steps: int,
    dt_ms: float = 0.5,
    drive_freq_hz: float = 1.5,
    seed: int = 42,
) -> np.ndarray:
    """Generate biologically-structured synthetic 302-neuron activity matrix.

    Returns X ∈ ℝ^{302 × n_steps} with k ≥ 4 independent principal components.

    This replaces the degenerate ``np.vstack([V_mus.T] * n)[:302]`` pattern used in
    earlier notebooks, which collapsed the 302-D space to rank 1 and prevented any
    NAML learning step from outperforming the Step 0 rule-based baseline.

    Neuron group structure (Boyle et al. 2012, White et al. 1986):

      | Group               | Count | Signal model                                |
      |---------------------|-------|---------------------------------------------|
      | Command interneurons| 12    | 4-phase rhythmic, strong amplitude          |
      | A-class MNs (VA, DA)| 21    | backward traveling wave                     |
      | B-class MNs (VB, DB)| 18    | forward traveling wave                      |
      | D-class MNs (VD, DD)| 19    | inhibitory antiphase                        |
      | Other interneurons  | 30    | multi-frequency oscillations (0.5f, f, 2f) |
      | Sensory neurons     | 100   | sparse exponential-decay bursts             |
      | Body / remaining    | 102   | slow oscillations + Gaussian noise          |
      | Total               | 302   |                                             |

    With k ≥ 4 independent PCs the NAML pipeline can:
    - Step 1 (SVD+Procrustes): align 4-D worm subspace to Chopin feature space
    - Step 2 (K-means): find 4-8 meaningful motor-state clusters
    - Steps 3-6 (Ridge/MLP/Adam/L-BFGS): learn a non-trivial mapping

    Parameters
    ----------
    n_steps       : number of simulation time steps (= duration_s * 1000 / dt_ms)
    dt_ms         : timestep in milliseconds (must match the forward model dt_ms)
    drive_freq_hz : locomotion drive frequency in Hz (default 1.5)
    seed          : random seed for reproducibility

    Returns
    -------
    X : np.ndarray of shape (302, n_steps), dtype float32
    """
    rng = np.random.default_rng(seed)
    omega = 2.0 * np.pi * drive_freq_hz * 1e-3  # rad/ms
    t = np.arange(n_steps, dtype=float) * dt_ms

    # ── Group 1: Command interneurons (12) — 4-phase rhythmic ──────────────
    # AVA, AVD (backward), AVB, PVC (forward), left/right copies → 4 × 3 = 12
    cmd_phases = [0.0, np.pi / 2, np.pi, 3.0 * np.pi / 2]
    X_cmd = np.zeros((12, n_steps))
    for i, ph in enumerate(cmd_phases):
        X_cmd[3 * i : 3 * i + 3] = (
            10.0 * np.sin(omega * t + ph)
            + rng.standard_normal((3, n_steps)) * 1.0
        )

    # ── Group 2: A-class motor neurons (21) — backward traveling wave ──────
    # VA1-VA12 (12) + DA1-DA9 (9) = 21; phase progresses head→tail
    X_mn_a = np.zeros((21, n_steps))
    for i in range(21):
        ph = np.pi + i * (2.0 * np.pi / 21.0)  # backward = antiphase to forward
        X_mn_a[i] = 7.0 * np.sin(omega * t + ph) + rng.standard_normal(n_steps) * 1.5

    # ── Group 3: B-class motor neurons (18) — forward traveling wave ───────
    # VB1-VB11 (11) + DB1-DB7 (7) = 18; phase progresses head→tail
    X_mn_b = np.zeros((18, n_steps))
    for i in range(18):
        ph = i * (2.0 * np.pi / 18.0)           # forward wave
        X_mn_b[i] = 7.0 * np.sin(omega * t + ph) + rng.standard_normal(n_steps) * 1.5

    # ── Group 4: D-class motor neurons (19) — inhibitory antiphase ─────────
    # VD1-VD13 (13) + DD1-DD6 (6) = 19; antiphase to B-class (GABA inhibition)
    X_mn_d = np.zeros((19, n_steps))
    for i in range(19):
        ph = np.pi + i * (2.0 * np.pi / 19.0)
        X_mn_d[i] = 5.0 * np.sin(omega * t + ph) + rng.standard_normal(n_steps) * 1.0

    # ── Group 5: Other interneurons (30) — multi-frequency ─────────────────
    # RIB, RIM, SMB, SMD, etc.: mixture of locomotion freq and harmonics
    X_intr = np.zeros((30, n_steps))
    freqs = [drive_freq_hz * 0.5, drive_freq_hz, drive_freq_hz * 2.0]
    for i in range(30):
        om_i = 2.0 * np.pi * freqs[i % 3] * 1e-3
        ph = rng.uniform(0.0, 2.0 * np.pi)
        X_intr[i] = 4.0 * np.sin(om_i * t + ph) + rng.standard_normal(n_steps) * 2.0

    # ── Group 6: Sensory neurons (100) — sparse exponential-decay bursts ───
    # Mechanosensory, chemosensory, proprioceptive: fire in sparse bursts
    X_sens = rng.standard_normal((100, n_steps)) * 0.5
    tau_decay = 20.0 / dt_ms   # 20 ms decay
    for i in range(100):
        n_bursts = rng.integers(5, 15)
        burst_times = rng.integers(0, n_steps, size=n_bursts)
        for bt in burst_times:
            dur = min(int(tau_decay * 3), n_steps - bt)
            if dur <= 0:
                continue
            X_sens[i, bt : bt + dur] += 3.0 * np.exp(-np.arange(dur) / tau_decay)

    # ── Group 7: Body / remaining (102) — slow oscillations + noise ────────
    X_body = rng.standard_normal((102, n_steps)) * 0.3
    om_slow = omega * 0.3   # slow proprioceptive oscillation
    for i in range(102):
        ph = rng.uniform(0.0, 2.0 * np.pi)
        X_body[i] += 1.5 * np.sin(om_slow * t + ph)

    # ── Assemble and verify ─────────────────────────────────────────────────
    X = np.vstack([X_cmd, X_mn_a, X_mn_b, X_mn_d, X_intr, X_sens, X_body])
    assert X.shape == (302, n_steps), f"Shape mismatch: {X.shape}"
    return X.astype(np.float32)


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
