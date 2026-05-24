"""Render and evaluation helpers for the Chopin score model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from pyannow.composer.piano_synth import synthesise_melody
from pyannow.targets.midi_target import musical_f1, pitch_aware_f1

from .data import roll_to_note_events


def score_to_events(
    times: np.ndarray,
    pitches: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    min_bins: int = 2,
):
    """Convert model probabilities to MIDI note events."""

    return roll_to_note_events(times=times, pitches=pitches, roll=probs, threshold=threshold, min_bins=min_bins)


def evaluate_reconstruction(
    predicted_events,
    target_events,
    duration_s: float,
):
    """Report onset and pitch-aware scores for the reconstructed score."""

    pred_onsets = np.array([ev.time_s for ev in predicted_events], dtype=float)
    target_onsets = np.array([ev.time_s for ev in target_events], dtype=float)
    f1 = musical_f1(pred_onsets, target_onsets, window_s=duration_s)

    pred_pitches = np.array([ev.pitch for ev in predicted_events], dtype=int)
    target_pitches = np.array([ev.pitch for ev in target_events], dtype=int)
    pf1 = pitch_aware_f1(
        pred_onsets,
        pred_pitches,
        target_onsets,
        target_pitches,
        window_s=duration_s,
        same_pitch_class=True,
    )

    return {"f1": f1, "pitch_f1": pf1}


def render_audio(events, duration_s: float, out_path: str | Path | None = None):
    """Synthesize audio and optionally save a WAV file."""

    audio, fs = synthesise_melody(events, duration_s=duration_s)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(str(out_path), fs, (np.asarray(audio) * 28000).astype(np.int16))
    return audio, fs
