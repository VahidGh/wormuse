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


# ─────────────────────────────────────────────────────────────────────────────
# Pitch-aware metrics  (ISSUE-035, ISSUE-020, ISSUE-021, ISSUE-037)  v0.8.0
# ─────────────────────────────────────────────────────────────────────────────

def pitch_aware_f1(
    worm_onsets:    np.ndarray,
    worm_pitches:   np.ndarray,
    chopin_onsets:  np.ndarray,
    chopin_pitches: np.ndarray,
    tol_s:          float = 0.05,
    window_s:       float = 15.0,
    same_pitch_class: bool = True,
) -> dict:
    """F1 requiring matches in BOTH time (±tol_s) AND pitch (exact or pitch-class).

    A worm note (t_w, p_w) matches a Chopin note (t_c, p_c) iff:
        |t_w − t_c| ≤ tol_s   AND   pitch_match(p_w, p_c)

    where pitch_match is exact MIDI equality when same_pitch_class=False,
    or same pitch-class (p mod 12) when same_pitch_class=True.

    The latter is pragmatic for the 96-cell Boyle model: each muscle fires a
    fixed chromatic pitch, so pitch-class matching gives credit when the worm
    hits the right note but in the wrong octave.

    Matching is greedy in time order with no double-claims (each Chopin note
    can be claimed by at most one worm note and vice versa).

    Parameters
    ----------
    worm_onsets    : (N,) sorted onset times in seconds
    worm_pitches   : (N,) MIDI pitch for each worm onset
    chopin_onsets  : (M,) sorted Chopin onset times in seconds
    chopin_pitches : (M,) MIDI pitch for each Chopin onset
    tol_s          : timing tolerance in seconds (default 50 ms)
    window_s       : analysis window (seconds)
    same_pitch_class : if True, match by pitch-class (p mod 12) instead of exact

    Returns
    -------
    dict: f1, precision, recall, tp, n_worm, n_chopin, pitch_acc
    """
    # Clip to window
    mask_w = worm_onsets   <= window_s
    mask_c = chopin_onsets <= window_s
    w_t = np.asarray(worm_onsets[mask_w],   dtype=float)
    w_p = np.asarray(worm_pitches[mask_w],  dtype=int)
    c_t = np.asarray(chopin_onsets[mask_c], dtype=float)
    c_p = np.asarray(chopin_pitches[mask_c], dtype=int)

    if len(w_t) == 0 or len(c_t) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                "tp": 0, "n_worm": int(len(w_t)), "n_chopin": int(len(c_t)),
                "pitch_acc": 0.0}

    def _pitch_match(p1: int, p2: int) -> bool:
        return (p1 % 12 == p2 % 12) if same_pitch_class else (p1 == p2)

    # Greedy one-to-one matching (sort by worm onset time)
    claimed_c = np.zeros(len(c_t), dtype=bool)
    claimed_w = np.zeros(len(w_t), dtype=bool)
    tp = 0

    order = np.argsort(w_t)
    for wi in order:
        t_w, p_w = w_t[wi], w_p[wi]
        # Find unclaimed Chopin notes within time tolerance
        cands = np.where(
            (~claimed_c) & (np.abs(c_t - t_w) <= tol_s)
        )[0]
        if len(cands) == 0:
            continue
        # Among candidates check pitch match
        pitch_ok = [ci for ci in cands if _pitch_match(p_w, c_p[ci])]
        if not pitch_ok:
            continue
        # Claim the closest-in-time pitch-matching Chopin note
        best = pitch_ok[np.argmin(np.abs(c_t[pitch_ok] - t_w))]
        claimed_c[best] = True
        claimed_w[wi]   = True
        tp += 1

    precision = tp / len(w_t)
    recall    = tp / len(c_t)
    f1 = (2.0 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # Pitch accuracy: among timing-only matches, fraction with correct pitch
    # (diagnostic — tells us if timing is right but pitch is wrong)
    tp_timing = int(sum(
        1 for wn in w_t if np.any(np.abs(c_t - wn) <= tol_s)
    ))
    pitch_acc = tp / max(1, tp_timing)

    return {
        "f1":        f1,
        "precision": precision,
        "recall":    recall,
        "tp":        tp,
        "n_worm":    int(len(w_t)),
        "n_chopin":  int(len(c_t)),
        "pitch_acc": pitch_acc,
    }


def biological_pitch_ceiling(
    muscle_pitches: np.ndarray,
    target_pitches: np.ndarray,
    same_pitch_class: bool = True,
) -> dict:
    """Fraction of Chopin pitches reachable by the worm's fixed muscle-pitch map.

    Each worm muscle cell fires a fixed MIDI pitch (MUSCLE_PITCHES_96).  A
    Chopin target pitch p_c is "reachable" iff there exists at least one muscle
    with pitch p_m such that pitch_match(p_m, p_c) is True.

    This is the **pitch ceiling** — the maximum pitch-aware recall the worm can
    ever achieve.  With the 96-cell Boyle model (MIDI 24-119, all 12 pitch
    classes × 8 octaves), every pitch class is covered so the pitch ceiling
    ≈ 100%.  With the legacy 8-cell model (8 fixed C#m pitches), ≈ 40% of
    Chopin pitches are unreachable.

    Parameters
    ----------
    muscle_pitches : MIDI pitch for each muscle (e.g. MUSCLE_PITCHES_96)
    target_pitches : MIDI pitches in the Chopin score
    same_pitch_class : if True, match by pitch-class (covers octave transpositions)

    Returns
    -------
    dict: reachable_fraction, reachable_N, total_N, unique_reachable_classes,
          unique_target_classes, n_muscles
    """
    target = np.asarray(target_pitches, dtype=int)
    muscles = np.asarray(muscle_pitches, dtype=int)

    if same_pitch_class:
        muscle_classes = set(int(p) % 12 for p in muscles)
        reachable = np.array([int(p) % 12 in muscle_classes for p in target])
    else:
        muscle_set = set(int(p) for p in muscles)
        reachable = np.array([int(p) in muscle_set for p in target])

    reachable_N = int(reachable.sum())
    total_N     = len(target)

    return {
        "reachable_fraction":      reachable_N / max(1, total_N),
        "reachable_N":             reachable_N,
        "total_N":                 total_N,
        "unique_reachable_classes": len(set(int(p) % 12 for p in target[reachable])),
        "unique_target_classes":    len(set(int(p) % 12 for p in target)),
        "n_muscles":               len(muscles),
    }


def precision_recall_at_tolerances(
    worm_onsets:   np.ndarray,
    target_onsets: np.ndarray,
    tols:          tuple = (0.025, 0.05, 0.10, 0.20),
    window_s:      float = 15.0,
) -> list[dict]:
    """Compute precision, recall, and F1 across multiple timing tolerances.

    Returns a list of dicts (one per tolerance) with keys:
        tol_s, precision, recall, f1

    Use this to build a multi-tolerance curve (AppStat L06: ROC/PR analogue).
    A robust onset detector should maintain F1 ≥ 0.20 even at tol_s=50 ms.
    """
    results = []
    for tol in tols:
        r = musical_f1(worm_onsets, target_onsets, tol_s=tol, window_s=window_s)
        results.append({
            "tol_s":     tol,
            "precision": r["precision"],
            "recall":    r["recall"],
            "f1":        r["f1"],
        })
    return results


def bootstrap_musical_f1(
    worm_onsets:   np.ndarray,
    target_onsets: np.ndarray,
    n_boot:        int   = 500,
    tol_s:         float = 0.05,
    window_s:      float = 15.0,
    ci:            float = 0.95,
    seed:          int   = 42,
) -> dict:
    """Bootstrap 95% CI for musical_f1 (AppStat L06 — bootstrap CIs).

    Strategy: resample Chopin note onsets with replacement (the target
    distribution), recompute F1 each time.  This gives a non-parametric CI
    that reflects the sensitivity of the score to which Chopin notes are
    evaluated.

    Parameters
    ----------
    n_boot : number of bootstrap replicates (default 500)
    ci     : confidence level (default 0.95)

    Returns
    -------
    dict: mean_f1, ci_low, ci_high, std_f1, n_boot
    """
    rng = np.random.default_rng(seed)
    t = target_onsets[target_onsets <= window_s]
    w = worm_onsets[worm_onsets <= window_s]

    if len(t) == 0 or len(w) == 0:
        return {"mean_f1": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "std_f1": 0.0, "n_boot": n_boot}

    f1s = []
    for _ in range(n_boot):
        t_boot = rng.choice(t, size=len(t), replace=True)
        r = musical_f1(w, np.sort(t_boot), tol_s=tol_s, window_s=window_s)
        f1s.append(r["f1"])

    f1s = np.array(f1s)
    alpha = (1.0 - ci) / 2.0
    return {
        "mean_f1": float(f1s.mean()),
        "ci_low":  float(np.quantile(f1s, alpha)),
        "ci_high": float(np.quantile(f1s, 1.0 - alpha)),
        "std_f1":  float(f1s.std()),
        "n_boot":  n_boot,
    }
