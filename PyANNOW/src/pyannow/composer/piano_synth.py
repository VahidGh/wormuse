"""Piano synthesiser — pure NumPy + optional SciPy reverb.

v1.0.0: ``render_string_v2`` — 3-string detuned choir + hammer-noise burst
  (ISSUE-003).  ``duration_s=None`` derives length from event stream (ISSUE-004).

v1.1.0: ``render_ks`` — Karplus-Strong physical wave-guide replaces the modal
  model as the default engine.  Correct piano timbre: long natural decay, pitch-
  dependent sustain, velocity-dependent brightness.  ``synthesise_from_hh`` adds
  a direct biophysics path: HH muscle-voltage spikes → KS hammer strikes, with
  no onset classifier needed.  ``synthesise_melody`` gains an ``engine`` keyword
  ("modal" | "v2" | "ks") and a ``None``-default ``note_sustain_s``.

Worm–piano biophysics analogy (v1.1.0):
  EGL-19 Ca²⁺ upswing  ≡  hammer impulse (excitation energy)
  EXP-2 K⁺ recovery    ≡  KS feedback decay g (string energy dissipation)
  NCA-1/2 Na⁺ leak      ≡  LP loss per cycle (bridge admittance / brightness)
  dV/dt at spike        ≡  hammer velocity  → MIDI velocity 1-127
  24 muscles / quadrant ≡  24 piano strings in one octave

Physics references:
  Karplus & Strong (1983) "Digital synthesis of plucked-string sounds"
  Chabassier, Chaigne & Joly (2014) "Time domain simulation of a piano"
"""
from __future__ import annotations

import numpy as np
from typing import Sequence

# ─── MIDI pitch → fundamental frequency ──────────────────────────────────────

def midi_to_hz(pitch: int) -> float:
    """MIDI pitch (0-127) → fundamental frequency in Hz."""
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


# ─── Default muscle→pitch maps ───────────────────────────────────────────────
try:
    from .worm_optimizer import MUSCLE_PITCHES, MUSCLE_PITCHES_95, generate_muscle_pitches
    WORM_PITCHES    = list(MUSCLE_PITCHES)
    WORM_PITCHES_95 = list(MUSCLE_PITCHES_95)
except Exception:
    WORM_PITCHES    = [61, 64, 66, 68, 71, 73, 76, 78]
    WORM_PITCHES_95 = WORM_PITCHES


# ─── Single-string modal synthesiser (v1 — kept for tests & backward compat) ─

def render_string(
    f0: float = 261.63,
    velocity: int = 64,
    duration_s: float = 0.8,
    fs: int = 22050,
    N_modes: int = 40,
    inharm_B: float = 4e-4,
    sigma0: float = 1.5,
    sigma1: float = 5e-5,
    x_hammer: float = 0.12,
    x_pickup: float = 0.50,
) -> np.ndarray:
    """Synthesise one struck piano string via modal synthesis (v1).

    Each mode n has frequency ``f_n = n·f0·√(1 + B·n²)`` (inharmonic from
    bending stiffness), decays with rate ``γ_n = σ0 + σ1·(2π·f_n)²``, and is
    excited proportionally to the hammer position (Chabassier/Bilbao).

    Parameters
    ----------
    f0         : fundamental frequency (Hz)
    velocity   : MIDI velocity 1-127 → hammer energy
    duration_s : note duration in seconds
    fs         : sample rate (Hz)
    N_modes    : number of partials (40 is perceptually complete)
    inharm_B   : inharmonicity coefficient
    sigma0     : frequency-independent damping
    sigma1     : frequency-dependent damping
    x_hammer   : normalised hammer position [0, 1]
    x_pickup   : normalised pickup position [0, 1]
    """
    amplitude = velocity / 127.0 * 0.8
    n_out = int(duration_s * fs)
    t     = np.arange(n_out, dtype=np.float64) / fs

    ns = np.arange(1, N_modes + 1, dtype=float)
    f_n     = ns * f0 * np.sqrt(1.0 + inharm_B * ns ** 2)
    gamma_n = sigma0 + sigma1 * (2.0 * np.pi * f_n) ** 2
    A_hammer = np.sin(ns * np.pi * x_hammer) / ns
    A_pickup = np.sin(ns * np.pi * x_pickup)

    phase   = 2.0 * np.pi * f_n[:, None] * t[None, :]
    decay   = np.exp(-gamma_n[:, None] * t[None, :])
    contrib = (A_hammer * A_pickup)[:, None] * decay * np.sin(phase)
    audio   = contrib.sum(axis=0) * amplitude

    peak = np.max(np.abs(audio))
    if peak > 1e-12:
        audio /= peak
        audio *= amplitude

    return audio.astype(np.float32)


# ─── Multi-string synthesiser with hammer transient (v2 — ISSUE-003) ─────────

def render_string_v2(
    f0: float = 261.63,
    velocity: int = 64,
    duration_s: float = 0.8,
    fs: int = 22050,
    N_modes: int = 40,
    inharm_B: float = 4e-4,
    sigma0: float = 1.5,
    sigma1: float = 5e-5,
    x_hammer: float = 0.12,
    x_pickup: float = 0.50,
    detune_cents: tuple[float, ...] = (0.0, +5.0, -5.0),
    hammer_noise_ms: float = 4.0,
) -> np.ndarray:
    """3-string detuned piano synthesiser with hammer-impact transient (v2).

    Improvements over ``render_string`` (ISSUE-003):

    * **3 strings** tuned at 0, +5, -5 cents — gives ~3 Hz beating at A4 that
      transforms the timbre from "tin can" to a recognisable piano choir.
    * **Hammer-noise burst** — 4 ms of band-limited white noise with a fast
      exponential decay models the physical impact of the hammer on the string.

    Parameters
    ----------
    detune_cents   : per-string pitch offsets in cents (len = n_strings)
    hammer_noise_ms: length of hammer-impact noise burst in milliseconds
    (remaining parameters identical to ``render_string``)
    """
    n_out = int(duration_s * fs)
    audio = np.zeros(n_out, dtype=np.float64)

    # 3 detuned strings — average so total power is comparable to v1
    for cents in detune_cents:
        f_str = f0 * (2.0 ** (cents / 1200.0))
        s = render_string(
            f0=f_str, velocity=velocity, duration_s=duration_s,
            fs=fs, N_modes=N_modes, inharm_B=inharm_B,
            sigma0=sigma0, sigma1=sigma1,
            x_hammer=x_hammer, x_pickup=x_pickup,
        ).astype(np.float64)
        audio[:len(s)] += s / len(detune_cents)

    # Hammer-impact burst: shaped white noise with 2 ms exponential decay
    n_noise = max(1, int(hammer_noise_ms * 1e-3 * fs))
    t_noise = np.arange(n_noise) / fs
    rng = np.random.default_rng(abs(hash(round(f0, 1))) % (2 ** 32))
    noise = rng.standard_normal(n_noise) * np.exp(-t_noise * (1.0 / 0.002))
    noise *= (velocity / 127.0) * 0.12
    audio[:n_noise] += noise

    peak = np.max(np.abs(audio))
    if peak > 1e-12:
        audio /= peak
        audio *= (velocity / 127.0 * 0.8)

    return audio.astype(np.float32)


# ─── Karplus-Strong physical wave-guide (v3 — default engine) ────────────────

def render_ks(
    f0: float = 261.63,
    velocity: int = 64,
    duration_s: float = 3.0,
    fs: int = 22050,
    decay_s: float | None = None,
    detune_cents: tuple[float, ...] = (0.0, +5.0, -5.0),
) -> np.ndarray:
    """Karplus-Strong struck-string synthesiser — piano-grade timbre (v3).

    Implements the recurrence ``y[n] = g/2·y[n-N] + g/2·y[n-N-1] + x[n]`` as
    a scipy IIR filter — O(N) per string, no Python sample loop.

    Why it sounds like a piano (and v1/v2 modal did not):

    * **Correct decay physics** — ``g`` derives from ``decay_s``, which auto-
      scales as ``5 s × (440/f0)^0.6``: bass notes ring for ~10 s, treble ~2 s.
      The ``sigma0=1.5`` modal constant (τ ≈ 0.67 s — drum territory) is gone.
    * **Velocity → brightness** — the excitation is LP-filtered white noise;
      the low-pass cutoff scales with velocity so pp sounds warm/round and ff
      sounds bright/percussive, matching real hammer-strike physics.
    * **3 detuned strings** produce the characteristic piano chorus/beating.

    Worm–piano biophysics analogy (see ``synthesise_from_hh``):

    +---------------------------------+-------------------------------+
    | Worm biophysics                 | KS piano model                |
    +=================================+===============================+
    | EGL-19 Ca²⁺ upswing            | Hammer impulse (excitation)   |
    | dV/dt at spike peak             | Hammer velocity → brightness  |
    | EXP-2 K⁺ recovery (repol.)     | Feedback decay factor g       |
    | NCA-1/2 Na⁺ leak (tonic dep.)  | LP loss per cycle (warmth)    |
    | 24 muscles per quadrant         | 24 strings in one octave      |
    +---------------------------------+-------------------------------+

    Parameters
    ----------
    f0           : fundamental frequency (Hz)
    velocity     : MIDI velocity 1-127
    duration_s   : render window; the string decays naturally within this window
    fs           : sample rate (Hz)
    decay_s      : 1/e energy sustain time; ``None`` → auto from f0
                   (~5 s at A4, scales as (440/f0)^0.6 — upright-piano fit)
    detune_cents : per-string pitch offsets (three strings → choir/beating)
    """
    from scipy.signal import lfilter

    vel_frac = velocity / 127.0
    n_out = int(duration_s * fs)
    audio = np.zeros(n_out, dtype=np.float64)

    for cents in detune_cents:
        f_str = float(np.clip(f0 * 2.0 ** (cents / 1200.0), 20.0, fs * 0.47))
        N = max(2, int(round(fs / f_str)))

        # ── auto decay: longer for lower pitches ──────────────────────────────
        if decay_s is None:
            t_decay = 5.0 * (440.0 / max(f_str, 55.0)) ** 0.6
        else:
            t_decay = float(decay_s)
        t_decay = float(np.clip(t_decay, 0.1, duration_s))
        # per-cycle gain: after T_decay·f_str cycles, energy = 1/e
        g = float(np.exp(-1.0 / (2.0 * t_decay * f_str)))

        # ── velocity-dependent LP-filtered noise excitation ───────────────────
        rng = np.random.default_rng(abs(hash(round(f_str, 2))) % (2 ** 32))
        noise = rng.standard_normal(N)
        # alpha=0.25 → heavy LP (pp, warm); alpha=0.90 → near-flat (ff, bright)
        alpha = 0.25 + 0.65 * vel_frac
        exc = lfilter([alpha], [1.0, -(1.0 - alpha)], noise)

        # ── KS IIR: y[n] - g/2·y[n-N] - g/2·y[n-N-1] = x[n] ────────────────
        a_ks = np.zeros(N + 2)
        a_ks[0]     = 1.0
        a_ks[N]     = -g / 2.0
        a_ks[N + 1] = -g / 2.0

        x_in = np.zeros(n_out)
        x_in[: min(N, n_out)] = exc[: min(N, n_out)]
        audio += lfilter(np.array([1.0]), a_ks, x_in) / len(detune_cents)

    peak = np.max(np.abs(audio))
    if peak > 1e-12:
        audio /= peak
    audio *= vel_frac * 0.8
    return audio.astype(np.float32)


# ─── Room impulse response ────────────────────────────────────────────────────

def _room_ir(fs: int, decay_s: float = 0.30, seed: int = 0) -> np.ndarray:
    """Simple synthetic room IR: direct + 3 early reflections + late reverb.

    RT60 ≈ ``decay_s``.  Generated entirely from NumPy so no extra deps.
    """
    n = int(decay_s * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    ir = rng.standard_normal(n) * np.exp(-t * 8.0)   # late diffuse field
    ir[0] = 1.0                                        # direct sound
    for delay_s, gain in [(0.008, 0.60), (0.016, 0.40), (0.028, 0.25)]:
        idx = int(delay_s * fs)
        if idx < n:
            ir[idx] += gain
    mx = np.abs(ir).max()
    return (ir / max(mx, 1e-12)).astype(np.float32)


# ─── Mix a list of note events into a mono audio array ───────────────────────

def synthesise_melody(
    events: Sequence,
    duration_s: float | None = None,
    fs: int = 22050,
    note_sustain_s: float | None = None,
    max_notes: int | None = None,
    pitch_map: "np.ndarray | list | None" = None,
    engine: str | None = None,
    use_v2: bool = True,
    reverb: bool = True,
    reverb_mix: float = 0.20,
    reverb_decay_s: float = 0.30,
) -> tuple[np.ndarray, int]:
    """Render a list of note events to a mono PCM array.

    ``events`` can be either:
    - ``NoteEvent`` objects from :func:`~pyannow.targets.midi_target.parse_midi`
    - ``(time_s, muscle_idx_or_pitch, velocity)`` tuples

    Parameters
    ----------
    events         : sequence of note events
    duration_s     : output audio length in seconds.  ``None`` → derived from
                     the last event time + 2 s (ISSUE-004 fix).
    fs             : sample rate
    note_sustain_s : per-note render window. ``None`` → 3.0 s for ``engine="ks"``
                     (natural KS decay), 0.7 s for modal/v2 (original default).
    max_notes      : if set, limit to first N notes
    pitch_map      : array mapping muscle index → MIDI pitch
    engine         : synthesis engine: ``"ks"`` (Karplus-Strong, default when
                     this arg is supplied), ``"v2"`` (3-string modal, v1.0.0
                     default), or ``"modal"`` (original single-string).
                     ``None`` falls back to the ``use_v2`` flag for backward
                     compatibility.
    use_v2         : kept for backward compatibility; ignored when ``engine``
                     is explicitly set
    reverb         : if True (default), apply a synthetic room impulse response
    reverb_mix     : wet/dry ratio for reverb (0 = dry, 1 = fully wet)
    reverb_decay_s : RT60 of the room model in seconds

    Returns
    -------
    audio : (N,) float32 normalised to ±1
    fs    : sample rate
    """
    # ── Resolve engine from use_v2 if not explicitly set ──────────────────────
    if engine is None:
        engine = "v2" if use_v2 else "modal"

    # ── Auto note_sustain_s: KS has natural decay so render longer ────────────
    if note_sustain_s is None:
        note_sustain_s = 3.0 if engine == "ks" else 0.7

    # ── Select renderer ───────────────────────────────────────────────────────
    if engine == "ks":
        _render = render_ks
    elif engine == "v2":
        _render = render_string_v2
    else:
        _render = render_string

    # ── Resolve duration from event stream if not supplied (ISSUE-004) ────────
    if duration_s is None:
        if events:
            times = [ev.time_s if hasattr(ev, "time_s") else float(ev[0]) for ev in events]
            duration_s = float(max(times)) + 2.0
        else:
            duration_s = 0.0

    n_total = int(duration_s * fs)
    out = np.zeros(n_total, dtype=np.float64)

    for i, ev in enumerate(events):
        if max_notes is not None and i >= max_notes:
            break

        if hasattr(ev, "time_s"):
            t_s, pitch, vel, dur = ev.time_s, ev.pitch, ev.velocity, ev.duration
        else:
            t_s, pitch_or_idx, vel = float(ev[0]), ev[1], int(ev[2])
            if pitch_or_idx > 12:
                pitch = pitch_or_idx
            else:
                if pitch_map is not None:
                    idx = int(pitch_or_idx) % len(pitch_map)
                    pitch = int(pitch_map[idx])
                else:
                    pitch = WORM_PITCHES[int(pitch_or_idx) % len(WORM_PITCHES)]
            dur = note_sustain_s

        if t_s > duration_s:
            break

        f0 = midi_to_hz(int(pitch))
        # KS: render the full sustain window (decay is handled by the IIR loop)
        # modal/v2: clamp to MIDI duration + 400 ms tail
        if engine == "ks":
            note_dur = float(note_sustain_s)
        else:
            note_dur = min(float(dur) + 0.4, note_sustain_s)
            note_dur = max(note_dur, 0.3)

        note_arr = _render(f0=f0, velocity=int(vel), duration_s=note_dur, fs=fs)

        i_start = int(t_s * fs)
        i_end   = min(i_start + len(note_arr), n_total)
        out[i_start:i_end] += note_arr[:i_end - i_start].astype(np.float64)

    # ── Optional room reverb ───────────────────────────────────────────────────
    if reverb and np.any(out != 0.0):
        from scipy.signal import fftconvolve
        ir  = _room_ir(fs, decay_s=reverb_decay_s).astype(np.float64)
        wet = fftconvolve(out, ir)[:n_total]
        out = out + reverb_mix * wet

    # ── Global normalise ───────────────────────────────────────────────────────
    peak = np.max(np.abs(out))
    if peak > 1e-12:
        out /= peak
    return out.astype(np.float32), fs


# ─── Direct biophysics path: HH voltage → KS synthesis ──────────────────────

def synthesise_from_hh(
    V_muscles: "np.ndarray",
    t_sim: "np.ndarray",
    pitch_map: "np.ndarray | list | None" = None,
    fs: int = 22050,
    v_thresh: float = -20.0,
    vel_scale: float | None = None,
    note_sustain_s: float = 3.0,
    reverb: bool = True,
    reverb_mix: float = 0.20,
    reverb_decay_s: float = 0.30,
) -> "tuple[np.ndarray, int]":
    """Synthesise piano audio directly from Hodgkin-Huxley muscle voltage traces.

    Each upward crossing of ``v_thresh`` (default −20 mV, near the EGL-19 Ca²⁺
    spike peak) in a muscle's voltage trace is treated as a hammer strike.
    Hammer velocity is proportional to ``|dV/dt|`` at the crossing — Ca²⁺ spike
    slope — so fast, strong spikes produce louder, brighter notes (``render_ks``
    velocity-dependent LP filter).  No onset classifier is needed.

    This provides a **biophysics path** that runs in parallel to the **NAML
    path** (RSVD → MLP onset classifier → ``synthesise_melody``).  Comparing
    the two synthesises is itself an experiment: the NAML path learns Chopin;
    the HH path plays what the worm's muscles actually do.

    Worm–piano biophysics correspondence
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    +---------------------------------+----------------------------------+
    | Worm biophysics                 | KS piano model                   |
    +=================================+==================================+
    | EGL-19 Ca²⁺ upswing (V(t))     | Hammer strikes string            |
    | |dV/dt| at spike peak           | Hammer velocity → note loudness  |
    | EXP-2 K⁺ recovery (repol.)     | Feedback decay g → sustain       |
    | NCA-1/2 Na⁺ leak (tonic dep.)  | LP warmth per string cycle       |
    | 24 muscles per quadrant         | 24 strings per octave voice      |
    | 4 quadrants DL / VL / DR / VR  | 4 octave bands (bass → treble)   |
    +---------------------------------+----------------------------------+

    Parameters
    ----------
    V_muscles     : (n_muscles, T_sim) HH membrane voltages in mV
    t_sim         : (T_sim,) simulation time axis in seconds
    pitch_map     : (n_muscles,) MIDI pitch per muscle; ``None`` → WORM_PITCHES_95
    fs            : output audio sample rate (Hz)
    v_thresh      : upward threshold in mV (−20 mV ≈ EGL-19 activation midpoint)
    vel_scale     : |dV/dt| (mV/s) → MIDI velocity scale factor.  ``None`` →
                    auto-calibrate: max spike maps to velocity 90
    note_sustain_s: KS render window per note (natural decay within this window)
    reverb        : apply synthetic room IR
    reverb_mix    : wet/dry mix
    reverb_decay_s: RT60 of the room model

    Returns
    -------
    audio : (N,) float32 normalised PCM
    fs    : sample rate
    """
    V_muscles = np.asarray(V_muscles)
    t_sim     = np.asarray(t_sim)
    n_muscles, T_sim = V_muscles.shape
    dt = float(t_sim[1] - t_sim[0]) if T_sim > 1 else 1e-3

    pm = list(pitch_map) if pitch_map is not None else WORM_PITCHES_95

    # ── detect upward threshold crossings → note events ───────────────────────
    raw_events: list[tuple[float, int, float]] = []
    for j in range(n_muscles):
        V     = V_muscles[j]
        pitch = int(pm[j % len(pm)])
        below = V[:-1] < v_thresh
        above = V[1:]  >= v_thresh
        for idx in np.nonzero(below & above)[0] + 1:
            t_s  = float(t_sim[min(int(idx), T_sim - 1)])
            i_lo = max(int(idx) - 1, 0)
            i_hi = min(int(idx) + 1, T_sim - 1)
            dvdt = (V[i_hi] - V[i_lo]) / max((i_hi - i_lo) * dt, 1e-12)
            raw_events.append((t_s, pitch, abs(dvdt)))

    if not raw_events:
        n_empty = max(1, int(float(t_sim[-1]) * fs)) if T_sim > 0 else fs
        return np.zeros(n_empty, dtype=np.float32), fs

    # ── auto-calibrate vel_scale: strongest spike → velocity 90 ──────────────
    if vel_scale is None:
        max_dvdt = max(e[2] for e in raw_events)
        vel_scale = 90.0 / max(max_dvdt, 1.0)

    scaled: list[tuple[float, int, int]] = sorted(
        (t_s, pitch, int(np.clip(dvdt * vel_scale, 1, 127)))
        for t_s, pitch, dvdt in raw_events
    )

    return synthesise_melody(
        scaled,
        duration_s=None,
        fs=fs,
        note_sustain_s=note_sustain_s,
        engine="ks",
        reverb=reverb,
        reverb_mix=reverb_mix,
        reverb_decay_s=reverb_decay_s,
    )


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    audio, _fs = synthesise_melody(
        [(0.0, 60, 80), (0.5, 64, 75), (1.0, 67, 70)],
        duration_s=3.0, fs=8000, engine="ks", reverb=True)
    print(f"3-note chord (ks+reverb, 8 kHz):  {time.perf_counter()-t0:.2f}s  "
          f"len={len(audio)}  peak={abs(audio).max():.3f}")

    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    T = 500
    t_arr = np.arange(T) * 0.002          # 2 ms steps → 1 s total
    V_fake = rng.uniform(-60, 20, (8, T)) # 8 muscles, random voltage
    audio_hh, _fs = synthesise_from_hh(V_fake, t_arr, fs=8000)
    print(f"HH path (8 muscles, 1 s sim):     {time.perf_counter()-t0:.2f}s  "
          f"len={len(audio_hh)}  peak={abs(audio_hh).max():.3f}")
