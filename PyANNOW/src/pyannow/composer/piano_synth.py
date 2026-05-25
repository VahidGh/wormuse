"""Piano synthesiser — pure NumPy + optional SciPy reverb.

v1.0.0: ``render_string_v2`` replaces the single-string modal model with a
3-string detuned unison choir and a 4 ms hammer-noise burst.  This removes the
"tin can" coloration (ISSUE-003).  ``synthesise_melody`` gains ``use_v2``,
``reverb``, and a ``duration_s=None`` default that derives length from the event
stream so the full Chopin piece renders without truncation (ISSUE-004).

The original ``render_string`` is kept for backward compatibility and tests.

Physics reference:
  Chabassier, Chaigne & Joly (2014)
  "Time domain simulation of a piano. Part 1: Model description"
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
    note_sustain_s: float = 0.7,
    max_notes: int | None = None,
    pitch_map: "np.ndarray | list | None" = None,
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
                     the last event time + 2 s (ISSUE-004 fix: renders the full
                     piece without an artificial cap).
    fs             : sample rate
    note_sustain_s : default note duration when event.duration is unavailable
    max_notes      : if set, limit to first N notes
    pitch_map      : array mapping muscle index → MIDI pitch
    use_v2         : if True (default), use ``render_string_v2`` (3 detuned
                     strings + hammer transient) — ISSUE-003 fix
    reverb         : if True (default), apply a synthetic room impulse response
    reverb_mix     : wet/dry ratio for reverb (0 = dry, 1 = fully wet)
    reverb_decay_s : RT60 of the room model in seconds

    Returns
    -------
    audio : (N,) float32 normalised to ±1
    fs    : sample rate
    """
    # ── Resolve duration from event stream if not supplied (ISSUE-004) ────────
    if duration_s is None:
        if events:
            times = [ev.time_s if hasattr(ev, "time_s") else float(ev[0]) for ev in events]
            duration_s = float(max(times)) + 2.0
        else:
            duration_s = 0.0

    n_total = int(duration_s * fs)
    out = np.zeros(n_total, dtype=np.float64)
    _render = render_string_v2 if use_v2 else render_string

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

        f0       = midi_to_hz(int(pitch))
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


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    audio, fs = synthesise_melody(
        [(0.0, 60, 80), (0.5, 64, 75), (1.0, 67, 70)],
        duration_s=3.0, fs=8000, use_v2=True, reverb=True)
    print(f"3-note chord (v2+reverb, 8 kHz): {time.perf_counter()-t0:.2f}s  "
          f"len={len(audio)}  peak={abs(audio).max():.3f}")
