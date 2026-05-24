"""Score extraction and event reconstruction helpers for the Chopin demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from pyannow.targets.midi_target import NoteEvent, parse_midi, piano_roll


@dataclass
class ScoreDataset:
    """Container for the score learning dataset."""

    events: list[NoteEvent]
    bpm: float
    pitches: np.ndarray
    times: np.ndarray
    features: np.ndarray
    target_roll: np.ndarray
    resolution_s: float
    duration_s: float


def fourier_time_features(
    times: np.ndarray,
    duration_s: float,
    bpm: float | None = None,
    n_harmonics: int = 12,
) -> np.ndarray:
    """Encode absolute time with sinusoidal features.

    The model learns a fixed score, so time is the key conditioning signal.
    The beat-phase terms help with phrase repetition and pulse alignment.
    """

    times = np.asarray(times, dtype=float)
    duration_s = max(float(duration_s), 1e-6)
    phase = times / duration_s

    feats = [phase[:, None]]
    for k in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * k * phase
        feats.append(np.sin(angle)[:, None])
        feats.append(np.cos(angle)[:, None])

    if bpm is not None and np.isfinite(bpm) and bpm > 0:
        beat_phase = times * float(bpm) / 60.0
        angle = 2.0 * np.pi * beat_phase
        feats.append(np.sin(angle)[:, None])
        feats.append(np.cos(angle)[:, None])

    return np.concatenate(feats, axis=1).astype(np.float32)


def build_score_dataset(
    midi_path: str | Path,
    resolution_s: float = 0.02,
    n_harmonics: int = 12,
) -> ScoreDataset:
    """Parse the Chopin MIDI file and build the time-to-roll learning set."""

    events, bpm = parse_midi(midi_path)
    pitches, times, roll = piano_roll(events, resolution_s=resolution_s)
    if len(times) == 0:
        raise ValueError("The MIDI file produced an empty piano roll")

    duration_s = float(times[-1])
    features = fourier_time_features(times, duration_s=duration_s, bpm=bpm, n_harmonics=n_harmonics)
    target_roll = roll.T.astype(np.float32)

    return ScoreDataset(
        events=events,
        bpm=float(bpm),
        pitches=np.asarray(pitches, dtype=int),
        times=np.asarray(times, dtype=np.float32),
        features=features,
        target_roll=target_roll,
        resolution_s=float(resolution_s),
        duration_s=duration_s,
    )


def _segments_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def roll_to_note_events(
    times: np.ndarray,
    pitches: np.ndarray,
    roll: np.ndarray,
    threshold: float = 0.5,
    min_bins: int = 2,
    velocity_floor: int = 35,
) -> list[NoteEvent]:
    """Convert a predicted piano-roll into MIDI note events."""

    times = np.asarray(times, dtype=float)
    pitches = np.asarray(pitches, dtype=int)
    roll = np.asarray(roll, dtype=float)
    if roll.ndim != 2:
        raise ValueError("roll must be 2-D with shape (T, P)")

    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.02
    events: list[NoteEvent] = []

    for pitch_idx, pitch in enumerate(pitches):
        mask = roll[:, pitch_idx] >= threshold
        for start, end in _segments_from_mask(mask):
            if end - start + 1 < min_bins:
                continue
            chunk = roll[start : end + 1, pitch_idx]
            velocity = int(np.clip(velocity_floor + 85.0 * float(chunk.max()), 1, 127))
            events.append(
                NoteEvent(
                    time_s=float(times[start]),
                    pitch=int(pitch),
                    velocity=velocity,
                    duration=float((end - start + 1) * dt),
                )
            )

    events.sort(key=lambda ev: (ev.time_s, ev.pitch))
    return events


def best_frame_threshold(
    probs: np.ndarray,
    target_roll: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Pick the frame threshold that maximises flattened F1."""

    probs = np.asarray(probs, dtype=float)
    target_roll = np.asarray(target_roll, dtype=int)
    grid = grid if grid is not None else np.linspace(0.10, 0.90, 33)

    best_thr = float(grid[0])
    best_f1 = -1.0
    y_true = target_roll.reshape(-1)

    for thr in grid:
        y_pred = (probs >= thr).astype(int).reshape(-1)
        score = float(f1_score(y_true, y_pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)

    return best_thr, best_f1
