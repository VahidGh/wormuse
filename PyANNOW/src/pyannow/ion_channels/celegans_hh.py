"""C. elegans-realistic ion channel models.

These differ fundamentally from the squid axon HH model used in the MVP
notebook. C. elegans neurons and muscles use a distinct channel repertoire:

  EGL-19  —  L-type voltage-gated Ca²⁺ (VGCC).  The primary driver of
             muscle action potentials. Tuned by the PINN (this is the key
             ion channel for musical output). Homologue: CACNA1S / Cav1.
  EXP-2   —  Delayed-rectifier K⁺ (Kv) channel. Repolarises the muscle,
             setting note duration. Homologue: KCNH / ERG.
  SHK-1   —  Shaw-type K⁺ channel. Repolarises motor neurons.
             Homologue: Kv3.
  NCA-1/2 —  Na⁺ leak channels (NALCN family). Provide a background
             depolarising current that sets neuron excitability.
  SHL-1   —  Transient A-type K⁺. Fast inactivating, shapes spike onset.
  UNC-2   —  P/Q-type VGCC. Presynaptic Ca²⁺ entry for neurotransmitter
             release. Parameter: governs the NMJ coupling strength.

References
----------
Bhatt, 2012; Jospin et al. 2002 (EGL-19); Fleischhauer et al. 2020 (UNC-2);
WormBook "Ion channels" chapter; the SC-PINN user project (naml-ion-channel-pinn).

Units: voltage in mV, time in ms, conductances in mS/cm², capacitance in μF/cm².
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────────────
# Reversal potentials (C. elegans muscle, from Jospin et al. 2002)
# ─────────────────────────────────────────────────────────────────────────────
E_Ca   =  60.0   # mV  (Nernst, [Ca²⁺]_out >> [Ca²⁺]_in)
E_K    = -80.0   # mV  (slightly more negative than squid)
E_Na   =  50.0   # mV
E_leak = -65.0   # mV  (set point for NCA leak)


@dataclass
class CelegansChannelParams:
    """Tunable ion-channel parameter set for a *C. elegans* muscle / motor neuron.

    The four most musically relevant parameters (PINN-tunable) are highlighted.
    All others are set to best-fit literature values and held fixed during
    the Chopin optimisation unless the user explicitly sweeps them.
    """
    # ── PINN-tunable (key musical parameters) ────────────────────────────
    g_EGL19:   float = 8.0    # mS/cm²  L-type Ca²⁺ maximal conductance
    V_half_Ca: float = -20.0  # mV      EGL-19 half-activation voltage
    tau_Ca:    float = 15.0   # ms      EGL-19 activation time constant
    g_EXP2:    float = 5.0    # mS/cm²  delayed-rectifier K⁺ (note duration)
    # ── Other tunable (less impactful on music) ───────────────────────────
    g_SHK1:    float = 3.0    # mS/cm²  Shaw K⁺ (motor neuron repolarisation)
    g_NCA:     float = 0.8    # mS/cm²  Na⁺ leak (baseline excitability)
    g_SHL1:    float = 2.0    # mS/cm²  transient K⁺ A-type
    g_UNC2:    float = 1.5    # mS/cm²  presynaptic P/Q Ca²⁺ (NMJ strength)
    # ── Membrane properties ───────────────────────────────────────────────
    C_m:       float = 1.0    # μF/cm²
    g_leak:    float = 0.05   # mS/cm²  passive leak

    def as_vector(self) -> np.ndarray:
        """Return the four PINN-tunable params as a 1-D array for the optimizer."""
        return np.array([self.g_EGL19, self.V_half_Ca, self.tau_Ca, self.g_EXP2])

    @classmethod
    def from_vector(cls, x: np.ndarray, base: "CelegansChannelParams | None" = None):
        """Reconstruct from a 4-element optimizer vector, keeping other fields."""
        b = base or cls()
        return cls(
            g_EGL19   = float(x[0]),
            V_half_Ca = float(x[1]),
            tau_Ca    = float(x[2]),
            g_EXP2    = float(x[3]),
            g_SHK1=b.g_SHK1, g_NCA=b.g_NCA,
            g_SHL1=b.g_SHL1, g_UNC2=b.g_UNC2,
            C_m=b.C_m, g_leak=b.g_leak,
        )

    BOUNDS = [   # (min, max) for each element of as_vector()
        (0.5,  25.0),   # g_EGL19
        (-50.0, 0.0),   # V_half_Ca
        (1.0,  80.0),   # tau_Ca
        (0.5,  20.0),   # g_EXP2
    ]


# Default realistic C. elegans params
DEFAULT_PARAMS = CelegansChannelParams()

# ─────────────────────────────────────────────────────────────────────────────
# EGL-19 : L-type Ca²⁺  (Jospin et al. 2002 fit)
# ─────────────────────────────────────────────────────────────────────────────

def egl19_inf(V: np.ndarray, p: CelegansChannelParams) -> np.ndarray:
    """Steady-state activation m_∞(V) — Boltzmann."""
    return 1.0 / (1.0 + np.exp(-(V - p.V_half_Ca) / 6.5))


def egl19_current(V, m_Ca, p: CelegansChannelParams) -> np.ndarray:
    """EGL-19 Ca²⁺ current (mA/cm²)."""
    return p.g_EGL19 * m_Ca * (V - E_Ca)


# ─────────────────────────────────────────────────────────────────────────────
# EXP-2 : delayed-rectifier K⁺  (adapted from Bhatt 2012)
# ─────────────────────────────────────────────────────────────────────────────

def exp2_inf(V: np.ndarray) -> np.ndarray:
    """Steady-state K⁺ activation n_∞(V)."""
    return 1.0 / (1.0 + np.exp(-(V + 15.0) / 12.0))


def exp2_tau(V: np.ndarray) -> np.ndarray:
    """Voltage-dependent time constant (ms)."""
    return 20.0 + 50.0 * np.exp(-((V + 15.0) / 25.0) ** 2)


def exp2_current(V, n_K, p: CelegansChannelParams) -> np.ndarray:
    return p.g_EXP2 * n_K ** 2 * (V - E_K)


# ─────────────────────────────────────────────────────────────────────────────
# NCA leak:  I_NCA = g_NCA * (V - E_Na)  — constant conductance
# ─────────────────────────────────────────────────────────────────────────────

def nca_current(V, p: CelegansChannelParams) -> np.ndarray:
    return p.g_NCA * (V - E_Na)


# ─────────────────────────────────────────────────────────────────────────────
# Passive leak
# ─────────────────────────────────────────────────────────────────────────────

def leak_current(V, p: CelegansChannelParams) -> np.ndarray:
    return p.g_leak * (V - E_leak)


# ─────────────────────────────────────────────────────────────────────────────
# Muscle cell model   (Morris-Lecar style: V, m_Ca, n_K)
# ─────────────────────────────────────────────────────────────────────────────

def muscle_rhs(t: float, state: np.ndarray,
               I_syn: float,
               p: CelegansChannelParams) -> np.ndarray:
    """ODE RHS for a single *C. elegans* body-wall muscle cell.

    state = [V, m_Ca, n_K]

    I_syn > 0 is excitatory (ACh from motor neurons), < 0 inhibitory (GABA).
    """
    V, m_Ca, n_K = state

    # Ion currents
    I_Ca   = egl19_current(V, m_Ca, p)
    I_K    = exp2_current(V, n_K, p)
    I_NCA  = nca_current(V, p)
    I_L    = leak_current(V, p)

    dV = (-I_Ca - I_K - I_NCA - I_L + I_syn) / p.C_m

    # Gating kinetics
    dm_Ca = (egl19_inf(V, p) - m_Ca) / p.tau_Ca
    tau_n  = exp2_tau(V)
    dn_K   = (exp2_inf(V) - n_K) / tau_n

    return np.array([dV, dm_Ca, dn_K])


def resting_state(p: CelegansChannelParams) -> np.ndarray:
    """Approximate resting state [V, m_Ca, n_K] — solve V_rest by current balance."""
    V_rest = -65.0   # mV — acceptable approximation for C. elegans muscle
    m_Ca_0 = egl19_inf(V_rest, p)
    n_K_0  = exp2_inf(V_rest)
    return np.array([V_rest, m_Ca_0, n_K_0])


# ─────────────────────────────────────────────────────────────────────────────
# Graded motor-neuron model  (5-state: V, m_Ca_MN, adapted Morris-Lecar)
# Uses UNC-2 (presynaptic Ca²⁺) and SHK-1 (K⁺ repolarisation)
# ─────────────────────────────────────────────────────────────────────────────

def motorneuron_rhs(t: float, state: np.ndarray,
                    I_cmd: float,
                    p: CelegansChannelParams) -> np.ndarray:
    """Simplified graded motor-neuron ODE.

    state = [V, m_Ca_MN]
    I_cmd = command interneuron synaptic drive (graded)
    """
    V, m_Ca_MN = state

    # UNC-2 (presynaptic Ca²⁺ ~ same functional form as EGL-19 but lower threshold)
    m_inf_mn = 1.0 / (1.0 + np.exp(-(V + 30.0) / 5.0))
    I_UNC2   = p.g_UNC2 * m_Ca_MN * (V - E_Ca)

    # SHK-1 instantaneous (fast, approximated as quasi-steady-state)
    n_shk = 1.0 / (1.0 + np.exp(-(V + 10.0) / 10.0))
    I_SHK1 = p.g_SHK1 * n_shk * (V - E_K)

    I_L  = p.g_leak * (V - E_leak)
    I_NCA = p.g_NCA * (V - E_Na)

    dV = (-I_UNC2 - I_SHK1 - I_L - I_NCA + I_cmd) / p.C_m
    dm = (m_inf_mn - m_Ca_MN) / 2.0   # fast UNC-2 activation (2 ms τ)
    return np.array([dV, dm])


def motorneuron_ntransmitter(V_mn: float, threshold: float = -50.0) -> float:
    """Graded neurotransmitter release (sigmoid of pre-synaptic V)."""
    return 1.0 / (1.0 + np.exp(-(V_mn - threshold) / 5.0))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: simulate a single muscle cell and return Ca²⁺ peaks (note events)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_muscle(I_syn_fn,
                    duration_ms: float = 1000.0,
                    p: CelegansChannelParams | None = None,
                    dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a single BWM cell.

    Parameters
    ----------
    I_syn_fn : callable  t → I_syn(t)
    duration_ms : simulation duration in ms
    p : channel params (default: DEFAULT_PARAMS)
    dt : timestep in ms

    Returns
    -------
    t_arr : (N,) time array in ms
    V_arr : (N,) membrane voltage
    """
    p = p or DEFAULT_PARAMS
    state = resting_state(p)
    t_arr = np.arange(0, duration_ms, dt)
    V_arr = np.zeros_like(t_arr)

    for k, t in enumerate(t_arr):
        I_s = I_syn_fn(t)
        rhs = muscle_rhs(t, state, I_s, p)
        state = state + dt * rhs          # Euler  (use small dt ≤ 0.1 ms)
        state[0] = np.clip(state[0], -90, 80)
        state[1:] = np.clip(state[1:], 0, 1)
        V_arr[k] = state[0]

    return t_arr, V_arr


def detect_muscle_peaks(t_arr: np.ndarray, V_arr: np.ndarray,
                        threshold: float = -30.0,
                        min_interval_ms: float = 50.0) -> np.ndarray:
    """Find upward-crossing of threshold in V(t) — each crossing = one 'note onset'.

    Refractory window of min_interval_ms prevents double-counting.
    """
    crossings = np.where((V_arr[:-1] < threshold) & (V_arr[1:] >= threshold))[0]
    peaks = t_arr[crossings]
    # Enforce refractory period
    if len(peaks) == 0:
        return peaks
    keep = [peaks[0]]
    for p in peaks[1:]:
        if p - keep[-1] >= min_interval_ms:
            keep.append(p)
    return np.array(keep)


# ─────────────────────────────────────────────────────────────────────────────
# Force → MIDI velocity helper
# ─────────────────────────────────────────────────────────────────────────────

def force_to_velocity(force_normalised: float, force_scale: float = 50.0) -> int:
    """Map normalised muscle force [0,1] → MIDI velocity [1,127].

    ``force_scale`` is the size-rescaling factor that bridges the micro-Newton
    forces a real *C. elegans* produces and a pianist's keystroke force.
    This is the biological amplification argument: the same ion-channel dynamics
    can drive either the worm's micro-scale or a piano-scale output.
    """
    return int(np.clip(force_normalised * force_scale, 1, 127))


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity helpers  (used in the notebook's ion-channel importance analysis)
# ─────────────────────────────────────────────────────────────────────────────

PARAM_NAMES = ["g_EGL19", "V_half_Ca", "tau_Ca", "g_EXP2"]
PARAM_LABELS = [
    "g_EGL19\n(Ca²⁺ conductance)",
    "V½_Ca\n(activation threshold)",
    "τ_Ca\n(activation speed)",
    "g_EXP2\n(K⁺ repolarisation)",
]
