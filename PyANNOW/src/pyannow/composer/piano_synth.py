"""Stiff-string piano synthesiser — pure NumPy, no external audio libraries.

Renders a list of NoteEvents (or (time_s, pitch, velocity) tuples) into a
mono PCM array at ``fs`` samples/second by simulating each piano string with
the 1-D wave equation (bending stiffness + two damping terms) from:

  Chabassier, Chaigne & Joly (2014)
  "Time domain simulation of a piano. Part 1: Model description"

This is the same model used in docs/scientific_foundation_demo.ipynb — now
packaged as a callable so both notebooks can use it.

Quick usage
-----------
::

    from pyannow.composer.piano_synth import synthesise_melody
    from pyannow.targets.midi_target import parse_midi, NoteEvent

    events, _ = parse_midi("shared/examples/chopin_nocturne_op_posth_csharp_minor.mid")
    audio, fs  = synthesise_melody(events[:50], duration_s=15.0)

    import scipy.io.wavfile as wf
    wf.write("chopin_synth.wav", fs, (audio * 32000).astype("int16"))
"""
from __future__ import annotations

import numpy as np
from typing import Sequence

# ─── MIDI pitch → fundamental frequency ──────────────────────────────────────

def midi_to_hz(pitch: int) -> float:
    """MIDI pitch (0-127) → fundamental frequency in Hz."""
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


# ─── Default muscle→pitch maps ───────────────────────────────────────────────
# Imported from worm_optimizer so they stay in sync.
# These are the fallback when no pitch_map is supplied directly.
try:
    from .worm_optimizer import MUSCLE_PITCHES, MUSCLE_PITCHES_95, generate_muscle_pitches
    WORM_PITCHES    = list(MUSCLE_PITCHES)       # 8-cell legacy
    WORM_PITCHES_95 = list(MUSCLE_PITCHES_95)    # 95-cell full BWM
except Exception:
    WORM_PITCHES    = [61, 64, 66, 68, 71, 73, 76, 78]
    WORM_PITCHES_95 = WORM_PITCHES


# ─── Single-string stiff-string FDM synthesiser ──────────────────────────────

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
    """Synthesise one struck piano string via **modal synthesis**.

    Each mode n has frequency ``f_n = n·f0·√(1 + B·n²)`` (inharmonic from
    bending stiffness), decays with rate ``γ_n = σ0 + σ1·(2π·f_n)²``, and
    is excited proportionally to the hammer position.  This is the standard
    Chabassier/Bilbao formulation and is unconditionally stable (no FDM grid).

    Parameters
    ----------
    f0         : fundamental frequency (Hz)
    velocity   : MIDI velocity 1-127 → hammer energy (amplitude scaling)
    duration_s : note duration in seconds
    fs         : sample rate (Hz)
    N_modes    : number of partials to include (40 is perceptually complete)
    inharm_B   : inharmonicity coefficient (see Scientific Foundation §B.2)
    sigma0     : frequency-independent damping (sets overall decay time)
    sigma1     : frequency-dependent damping (higher modes decay faster)
    x_hammer   : normalised hammer position [0,1] along the string
    x_pickup   : normalised pickup (listener) position [0,1]
    """
    amplitude = velocity / 127.0 * 0.8   # normalised strike energy
    n_out = int(duration_s * fs)
    t     = np.arange(n_out, dtype=np.float64) / fs
    audio = np.zeros(n_out, dtype=np.float64)

    ns = np.arange(1, N_modes + 1, dtype=float)

    # Modal frequencies (inharmonic — eq. B.2 in SCIENTIFIC_FOUNDATION.md)
    f_n = ns * f0 * np.sqrt(1.0 + inharm_B * ns ** 2)

    # Per-mode damping rates (Chabassier energy model)
    gamma_n = sigma0 + sigma1 * (2.0 * np.pi * f_n) ** 2

    # Hammer excitation amplitude: sin(n·π·x_h) weighted by 1/n
    # (from the string's Green's function at the hammer point)
    A_hammer = np.sin(ns * np.pi * x_hammer) / ns

    # Pickup coupling: sin(n·π·x_p)
    A_pickup = np.sin(ns * np.pi * x_pickup)

    # Combine: for each mode, add a damped sinusoid
    # Vectorised over modes: (N_modes, n_out) → sum over modes → (n_out,)
    phase = 2.0 * np.pi * f_n[:, None] * t[None, :]         # (M, N)
    decay = np.exp(-gamma_n[:, None] * t[None, :])           # (M, N)
    contrib = (A_hammer * A_pickup)[:, None] * decay * np.sin(phase)  # (M, N)
    audio = contrib.sum(axis=0) * amplitude

    # Normalise (preserve relative dynamics per velocity)
    peak = np.max(np.abs(audio))
    if peak > 1e-12:
        audio /= peak
        audio *= amplitude

    return audio.astype(np.float32)


# ─── Mix a list of note events into a stereo/mono audio array ────────────────

def synthesise_melody(
    events: Sequence,
    duration_s: float = 15.0,
    fs: int = 22050,
    note_sustain_s: float = 0.7,
    max_notes: int | None = None,
    pitch_map: "np.ndarray | list | None" = None,
) -> tuple[np.ndarray, int]:
    """Render a list of note events to a mono PCM array.

    ``events`` can be either:
    - ``NoteEvent`` objects from :func:`~pyannow.targets.midi_target.parse_midi`
    - ``(time_s, muscle_idx_or_pitch, velocity)`` tuples (from worm forward model)

    Parameters
    ----------
    events         : sequence of note events
    duration_s     : output audio length in seconds
    fs             : sample rate
    note_sustain_s : default note duration (when event.duration unavailable)
    max_notes      : if set, limit to first N notes (for speed)
    pitch_map      : optional array mapping muscle index → MIDI pitch.
                     Pass ``result['pitch_map']`` from :func:`run_forward_fast`
                     to correctly translate 95-cell muscle indices to pitches.
                     If None, falls back to the 8-cell ``WORM_PITCHES`` list.

    Returns
    -------
    audio : (N,) float32 array normalised to ±1
    fs    : sample rate
    """
    out = np.zeros(int(duration_s * fs), dtype=np.float64)

    for i, ev in enumerate(events):
        if max_notes is not None and i >= max_notes:
            break

        # Handle both NoteEvent objects and (time_s, pitch_or_idx, vel) tuples
        if hasattr(ev, "time_s"):
            t_s, pitch, vel, dur = ev.time_s, ev.pitch, ev.velocity, ev.duration
        else:
            t_s, pitch_or_idx, vel = float(ev[0]), ev[1], int(ev[2])
            if pitch_or_idx > 12:
                # Already a MIDI pitch number
                pitch = pitch_or_idx
            else:
                # Muscle index — look up in pitch_map
                if pitch_map is not None:
                    idx = int(pitch_or_idx) % len(pitch_map)
                    pitch = int(pitch_map[idx])
                else:
                    # Fallback: use legacy 8-cell table
                    pitch = WORM_PITCHES[int(pitch_or_idx) % len(WORM_PITCHES)]
            dur = note_sustain_s

        if t_s > duration_s:
            break

        f0        = midi_to_hz(int(pitch))
        note_dur  = min(float(dur) + 0.4, note_sustain_s)   # string decay
        note_dur  = max(note_dur, 0.3)
        note_arr  = render_string(f0=f0, velocity=int(vel), duration_s=note_dur, fs=fs)

        i_start = int(t_s * fs)
        i_end   = min(i_start + len(note_arr), len(out))
        out[i_start:i_end] += note_arr[:i_end - i_start].astype(np.float64)

    # Global normalise
    peak = np.max(np.abs(out))
    if peak > 1e-12:
        out /= peak
    return out.astype(np.float32), fs


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    audio, fs = render_string(f0=261.63, velocity=80, duration_s=1.0), 22050
    print(f"C4 rendered in {time.perf_counter()-t0:.2f}s  "
          f"len={len(audio[0])}")
