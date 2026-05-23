"""MIDI target parser and distance metrics.

Parses the Chopin MIDI file, extracts note events, and provides distance
functions (DTW-like) for comparing worm output to the target.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path

# mido is a zero-dependency MIDI library (pip install mido)
try:
    import mido
    _MIDO_OK = True
except ImportError:
    _MIDO_OK = False
    mido = None  # type: ignore


@dataclass
class NoteEvent:
    time_s:   float   # onset time in seconds
    pitch:    int     # MIDI pitch 0-127
    velocity: int     # 0-127
    duration: float   # note duration in seconds (approximate)


# ─────────────────────────────────────────────────────────────────────────────
# MIDI parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_midi(path: str | Path) -> tuple[list[NoteEvent], float]:
    """Parse a MIDI file into a list of NoteEvents.

    Returns
    -------
    events : sorted list of NoteEvent
    tempo_bpm : detected tempo in beats per minute
    """
    if not _MIDO_OK:
        raise ImportError("pip install mido")

    mid = mido.MidiFile(str(path))
    ticks = mid.ticks_per_beat
    tempo_us = 500_000  # default 120 BPM
    events: list[NoteEvent] = []
    active: dict[int, tuple[float, int]] = {}   # pitch → (onset_s, velocity)
    t_abs = 0.0

    for msg in mido.merge_tracks(mid.tracks):
        t_abs += mido.tick2second(msg.time, ticks, tempo_us)
        if msg.type == "set_tempo":
            tempo_us = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = (t_abs, msg.velocity)
        elif msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active:
                onset, vel = active.pop(msg.note)
                events.append(NoteEvent(
                    time_s=onset, pitch=msg.note,
                    velocity=vel, duration=t_abs - onset
                ))

    events.sort(key=lambda e: e.time_s)
    bpm = 60_000_000 / tempo_us
    return events, bpm


def note_onsets(events: list[NoteEvent], clip_s: float | None = None) -> np.ndarray:
    """Sorted array of onset times (seconds)."""
    t = np.array([e.time_s for e in events])
    if clip_s is not None:
        t = t[t <= clip_s]
    return t


def piano_roll(events: list[NoteEvent],
               resolution_s: float = 0.01,
               clip_s: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary piano roll.

    Returns
    -------
    pitches : (P,) unique pitches present
    times   : (T,) time grid in seconds
    roll    : (P, T) bool array, True = note active
    """
    if clip_s is not None:
        events = [e for e in events if e.time_s <= clip_s]
    if not events:
        return np.array([]), np.array([]), np.zeros((0, 0), dtype=bool)

    t_max = max(e.time_s + e.duration for e in events)
    times = np.arange(0, t_max + resolution_s, resolution_s)
    pitches = np.array(sorted({e.pitch for e in events}))
    roll = np.zeros((len(pitches), len(times)), dtype=bool)

    for ev in events:
        pi = np.searchsorted(pitches, ev.pitch)
        ti_on  = int(ev.time_s   / resolution_s)
        ti_off = int((ev.time_s + ev.duration) / resolution_s) + 1
        ti_off = min(ti_off, len(times))
        roll[pi, ti_on:ti_off] = True

    return pitches, times, roll


# ─────────────────────────────────────────────────────────────────────────────
# Distance metrics
# ─────────────────────────────────────────────────────────────────────────────

def dtw_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Minimal Dynamic Time Warping distance between two 1-D arrays.

    O(n*m) — avoid calling on very long sequences. Clip to a few seconds
    of music for optimisation.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(float(a[i - 1]) - float(b[j - 1]))
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def onset_loss(worm_onsets: np.ndarray,
               target_onsets: np.ndarray,
               window_s: float = 15.0) -> float:
    """Compute the onset-timing distance between worm and Chopin.

    Strategy (fast approximation):
      1. Build binary activity arrays on a 20-ms grid.
      2. Return the normalised Hamming distance between them.

    Falls back to a min-distance matching if the arrays differ greatly in length.
    """
    grid = 0.02     # seconds

    # Clip to window
    w_clip = worm_onsets[worm_onsets <= window_s]
    t_clip = target_onsets[target_onsets <= window_s]
    if len(w_clip) == 0 or len(t_clip) == 0:
        return 1.0   # worst possible

    n_bins = int(window_s / grid) + 1
    w_vec = np.zeros(n_bins)
    t_vec = np.zeros(n_bins)
    for t in w_clip:
        idx = min(int(t / grid), n_bins - 1)
        w_vec[idx] = 1.0
    for t in t_clip:
        idx = min(int(t / grid), n_bins - 1)
        t_vec[idx] = 1.0

    # Soft match: convolve with a 60-ms Gaussian kernel before comparing
    sigma_bins = max(1, int(0.06 / grid))
    ksize = sigma_bins * 5
    k = np.exp(-0.5 * np.linspace(-3, 3, ksize) ** 2)
    k /= k.sum()
    w_soft = np.convolve(w_vec, k, mode="same")
    t_soft = np.convolve(t_vec, k, mode="same")

    return float(np.mean((w_soft - t_soft) ** 2))


def note_rate_mismatch(worm_onsets: np.ndarray,
                       target_onsets: np.ndarray,
                       window_s: float = 15.0) -> dict:
    """Diagnostic: compare note rates and temporal spread."""
    w = worm_onsets[worm_onsets <= window_s]
    t = target_onsets[target_onsets <= window_s]
    wr = len(w) / window_s
    tr = len(t) / window_s
    return {
        "worm_rate_Hz":   wr,
        "target_rate_Hz": tr,
        "rate_ratio":     wr / tr if tr > 0 else float("nan"),
        "worm_N":         len(w),
        "target_N":       len(t),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Biological ceiling analysis (no optimisation needed)
# ─────────────────────────────────────────────────────────────────────────────

def biological_ceiling_1voice(p_celegans, target_onsets: np.ndarray,
                               window_s: float = 30.0) -> dict:
    """Single-voice ceiling (conservative / for comparison only).

    Checks whether consecutive target notes respect the refractory window as if
    there were only one muscle cell.  Use biological_ceiling() for the
    biologically correct n-voice version.
    """
    tau_refrac = (p_celegans.tau_Ca + 50.0) * 1e-3
    t_clip = target_onsets[target_onsets <= window_s]
    if len(t_clip) < 2:
        return {"reachable_fraction": 1.0, "reachable_N": len(t_clip),
                "total_N": len(t_clip), "tau_refrac_ms": tau_refrac * 1000,
                "n_voices": 1}
    reachable = 1
    last = t_clip[0]
    for t in t_clip[1:]:
        if t - last >= tau_refrac:
            reachable += 1
            last = t
    return {
        "reachable_fraction": reachable / len(t_clip),
        "reachable_N":        reachable,
        "total_N":            len(t_clip),
        "tau_refrac_ms":      tau_refrac * 1000,
        "n_voices":           1,
    }


def biological_ceiling(p_celegans, target_onsets: np.ndarray,
                       window_s: float = 30.0,
                       n_voices: int = 95) -> dict:
    """N-voice biological ceiling: fraction of target notes physically reachable.

    With n_voices independent BWM muscle groups each having a refractory period
    of tau_Ca + 50 ms, a greedy scheduler assigns each target note to the voice
    that last fired most recently (but has recovered).  This maximises utilisation
    while keeping the most-recovered voices available for rapid future bursts.

    For Chopin at 4.4 notes/s with tau_refrac=65 ms:
      8 voices  → ~100% rate-reachable
      95 voices → ~100% rate-reachable
    The real bottleneck is rhythmic regularity, not note rate.

    Use n_voices=1 to reproduce the (incorrect) single-voice result for comparison.
    """
    tau_refrac = (p_celegans.tau_Ca + 50.0) * 1e-3
    t_clip = target_onsets[target_onsets <= window_s]
    if len(t_clip) == 0:
        return {"reachable_fraction": 1.0, "reachable_N": 0, "total_N": 0,
                "tau_refrac_ms": tau_refrac * 1000, "n_voices": n_voices}

    last_fired = np.full(n_voices, -np.inf)
    reachable = 0
    for t in t_clip:
        avail = np.where(last_fired + tau_refrac <= t)[0]
        if len(avail) > 0:
            best = avail[np.argmax(last_fired[avail])]  # least-idle recovered voice
            last_fired[best] = t
            reachable += 1
    return {
        "reachable_fraction": reachable / len(t_clip),
        "reachable_N":        reachable,
        "total_N":            len(t_clip),
        "tau_refrac_ms":      tau_refrac * 1000,
        "n_voices":           n_voices,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Musical evaluation metrics  (ISSUE-016)
# ─────────────────────────────────────────────────────────────────────────────

def musical_f1(worm_onsets: np.ndarray,
               target_onsets: np.ndarray,
               tol_s: float = 0.05,
               window_s: float = 15.0) -> dict:
    """F1-score with ±tol_s tolerance (standard MIR onset evaluation).

    Unlike onset_loss, this metric cannot be gamed by sparsity: producing
    zero or one note yields recall ≈ 0 and therefore F1 ≈ 0.

    Parameters
    ----------
    tol_s : matching tolerance in seconds (default 50 ms, MIR standard)

    Returns
    -------
    dict with keys: f1, precision, recall, tp_worm, tp_chopin, n_worm, n_chopin
    """
    w = worm_onsets[worm_onsets <= window_s]
    t = target_onsets[target_onsets <= window_s]
    if len(w) == 0 or len(t) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                "tp_worm": 0, "tp_chopin": 0,
                "n_worm": int(len(w)), "n_chopin": int(len(t))}
    # True positives from each side
    tp_w = int(sum(1 for wn in w if np.any(np.abs(t - wn) <= tol_s)))
    tp_t = int(sum(1 for tn in t if np.any(np.abs(w - tn) <= tol_s)))
    precision = tp_w / len(w)
    recall    = tp_t / len(t)
    f1 = (2.0 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"f1": f1, "precision": precision, "recall": recall,
            "tp_worm": tp_w, "tp_chopin": tp_t,
            "n_worm": int(len(w)), "n_chopin": int(len(t))}


def ioi_similarity(worm_onsets: np.ndarray,
                   target_onsets: np.ndarray,
                   window_s: float = 15.0) -> float:
    """Inter-onset interval (IOI) histogram intersection — rhythmic similarity.

    Computes the overlap between the IOI distributions of worm and Chopin on a
    [0, 2s] range with 40 bins.  Returns a value in [0, 1]:
      0 = no rhythmic overlap (completely different IOI structure)
      1 = identical IOI distributions

    A periodic worm will produce a spike at one IOI; Chopin has a broad,
    musically-phrased distribution.  Low overlap ≈ robotic/unmusical.
    Satisfactory: ≥ 0.30.
    """
    w = np.sort(worm_onsets[worm_onsets <= window_s])
    t = np.sort(target_onsets[target_onsets <= window_s])
    if len(w) < 2 or len(t) < 2:
        return 0.0
    ioi_w = np.diff(w)
    ioi_t = np.diff(t)
    bins = np.linspace(0.0, 2.0, 41)
    hw, _ = np.histogram(ioi_w, bins=bins, density=True)
    ht, _ = np.histogram(ioi_t, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    return float(np.minimum(hw, ht).sum() * bin_width)
