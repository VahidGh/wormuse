"""Scoring functions — AppStat 2026 Lecture 06 / Lab VI.

Lecture recap
-------------
A scalar F1 at one tolerance is a single operating point.  The full
diagnostic is:

- **F1 vs tolerance** curve — how does F1 change as we widen the matching
  window?  Robust steps maintain high F1 across tolerances; brittle ones
  collapse at small tolerance.
- **PR curve** — precision vs recall across thresholds; AUC-PR is the
  preferred summary on heavy imbalance.
- **ROC curve** — TPR vs FPR; AUC-ROC is the threshold-free quality.
- **Bootstrap CIs** — resample the test windows to put a 95% CI around
  each step's F1; otherwise rank claims like "Step k > Step j" have no
  statistical content.

This module wraps `pyannow.targets.midi_target.musical_f1` (no need to
re-implement — that function already exists from PyANNOW ISSUE-016) and
adds the AppStat extensions covered by ISSUE-020 and ISSUE-021.

Composite scoring (the goal-driven part)
----------------------------------------
The wormuse goal is "best worm→Chopin model" — a model is judged by a
vector of scores, not a scalar:

    pitch_aware_f1          — onsets must match in BOTH time AND pitch
    AUC-PR                  — threshold-free quality of the activation
    ioi_similarity          — rhythmic distribution overlap (from pyannow)
    velocity_correlation    — dynamics match (Pearson r over matched notes)
    bootstrap 95% CI        — uncertainty around every claim

These five compose the "Chopin-likeness" score used in pipeline.py.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd


def f1_vs_tolerance(worm_onsets: np.ndarray, chopin_onsets: np.ndarray,
                    tols_s: Iterable[float] = (0.010, 0.025, 0.050, 0.100, 0.200, 0.400),
                    window_s: float = 10.0) -> pd.DataFrame:
    """ISSUE-020: F1 / precision / recall as a function of matching tolerance.

    Calls `pyannow.targets.midi_target.musical_f1` at each tolerance; the
    PyANNOW function is the source of truth for the F1 definition.
    """
    from pyannow.targets.midi_target import musical_f1

    rows = []
    for tol in tols_s:
        r = musical_f1(worm_onsets, chopin_onsets, tol_s=tol, window_s=window_s)
        rows.append({"tol_s": tol, **r})
    return pd.DataFrame(rows)


def bootstrap_f1(worm_onsets: np.ndarray, chopin_onsets: np.ndarray,
                  B: int = 1000, window_s: float = 10.0, sub_window_s: float = 5.0,
                  tol_s: float = 0.05, random_state: int = 0) -> dict:
    """ISSUE-021: bootstrap CIs by resampling Chopin sub-windows.

    For each of B replicates, sample a random sub-window of length
    `sub_window_s` from [0, window_s], compute F1 within it, then return
    the 2.5 / 50 / 97.5 percentiles.  This is the AppStat-flavoured way
    to put a confidence interval on a single per-step F1 number.
    """
    from pyannow.targets.midi_target import musical_f1

    rng = np.random.default_rng(random_state)
    f1s = []
    for _ in range(B):
        t0 = rng.uniform(0.0, max(0.0, window_s - sub_window_s))
        t1 = t0 + sub_window_s
        w = worm_onsets[(worm_onsets >= t0) & (worm_onsets <= t1)] - t0
        c = chopin_onsets[(chopin_onsets >= t0) & (chopin_onsets <= t1)] - t0
        if len(w) == 0 or len(c) == 0:
            f1s.append(0.0)
            continue
        f1s.append(musical_f1(w, c, tol_s=tol_s, window_s=sub_window_s)["f1"])
    f1s = np.array(f1s)
    return {
        "median":  float(np.median(f1s)),
        "ci_low":  float(np.percentile(f1s, 2.5)),
        "ci_high": float(np.percentile(f1s, 97.5)),
        "samples": f1s,
    }


def pitch_aware_f1(worm_onsets: np.ndarray, worm_pitches: np.ndarray,
                    chopin_onsets: np.ndarray, chopin_pitches: np.ndarray,
                    tol_s: float = 0.05, window_s: float = 10.0,
                    same_pitch_class: bool = True) -> dict:
    """ISSUE-035: F1 that requires onsets match in BOTH time and pitch.

    A worm onset at time t with pitch p is a true positive only if there
    exists a Chopin onset at time t' with pitch p' such that |t - t'| <= tol_s
    AND (p == p' or, if same_pitch_class, p mod 12 == p' mod 12).

    This is the scoring function the worm→Chopin task actually requires.
    Plain musical_f1 (PyANNOW ISSUE-016) ignores pitch — a worm that gets the
    rhythm right but plays random pitches scores full marks.

    Greedy bipartite matching (good enough at these scales):
    sort by time, match each worm onset to its nearest unclaimed Chopin onset
    within tol_s AND same pitch class.
    """
    w_clip_mask = worm_onsets <= window_s
    t_clip_mask = chopin_onsets <= window_s
    w_on = np.asarray(worm_onsets)[w_clip_mask]
    w_p  = np.asarray(worm_pitches)[w_clip_mask]
    t_on = np.asarray(chopin_onsets)[t_clip_mask]
    t_p  = np.asarray(chopin_pitches)[t_clip_mask]

    if len(w_on) == 0 or len(t_on) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                "tp": 0, "n_worm": int(len(w_on)), "n_chopin": int(len(t_on))}

    # Greedy match — sort worm by time, walk Chopin candidates
    claimed = np.zeros(len(t_on), dtype=bool)
    tp = 0
    for wt, wp in zip(w_on, w_p):
        # Candidate Chopin onsets within tol_s and matching pitch class
        if same_pitch_class:
            cand = np.where((np.abs(t_on - wt) <= tol_s) &
                            ((t_p % 12) == (int(wp) % 12)) &
                            (~claimed))[0]
        else:
            cand = np.where((np.abs(t_on - wt) <= tol_s) &
                            (t_p == wp) & (~claimed))[0]
        if len(cand) == 0:
            continue
        # Pick the closest in time
        best = cand[np.argmin(np.abs(t_on[cand] - wt))]
        claimed[best] = True
        tp += 1

    prec = tp / len(w_on)
    rec  = tp / len(t_on)
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"f1": float(f1), "precision": float(prec), "recall": float(rec),
            "tp": int(tp), "n_worm": int(len(w_on)), "n_chopin": int(len(t_on))}


def velocity_correlation(worm_onsets: np.ndarray, worm_vel: np.ndarray,
                          chopin_onsets: np.ndarray, chopin_vel: np.ndarray,
                          tol_s: float = 0.05) -> float:
    """ISSUE-035: Pearson r between aligned-note velocities.

    Greedy time-match worm onsets to Chopin onsets within tol_s; report
    Pearson r over (worm_vel[matched], chopin_vel[matched]).  Returns NaN
    if fewer than 3 matches.  Measures *dynamics* — soft vs loud notes.
    """
    w_on = np.asarray(worm_onsets)
    w_v  = np.asarray(worm_vel, dtype=float)
    t_on = np.asarray(chopin_onsets)
    t_v  = np.asarray(chopin_vel, dtype=float)
    if len(w_on) == 0 or len(t_on) == 0:
        return float("nan")

    claimed = np.zeros(len(t_on), dtype=bool)
    pairs = []
    for wt, wv in zip(w_on, w_v):
        cand = np.where((np.abs(t_on - wt) <= tol_s) & (~claimed))[0]
        if len(cand) == 0:
            continue
        best = cand[np.argmin(np.abs(t_on[cand] - wt))]
        claimed[best] = True
        pairs.append((wv, t_v[best]))
    if len(pairs) < 3:
        return float("nan")
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_pitch_aware_f1(worm_onsets: np.ndarray, worm_pitches: np.ndarray,
                              chopin_onsets: np.ndarray, chopin_pitches: np.ndarray,
                              B: int = 400, window_s: float = 10.0,
                              sub_window_s: float = 5.0, tol_s: float = 0.05,
                              random_state: int = 0) -> dict:
    """Bootstrap CI on pitch_aware_f1 — closes ISSUE-021 for the pitch metric."""
    rng = np.random.default_rng(random_state)
    f1s = []
    for _ in range(B):
        t0 = rng.uniform(0.0, max(0.0, window_s - sub_window_s))
        t1 = t0 + sub_window_s
        w_idx = (worm_onsets >= t0) & (worm_onsets <= t1)
        c_idx = (chopin_onsets >= t0) & (chopin_onsets <= t1)
        if not w_idx.any() or not c_idx.any():
            f1s.append(0.0)
            continue
        r = pitch_aware_f1(
            worm_onsets[w_idx] - t0, worm_pitches[w_idx],
            chopin_onsets[c_idx] - t0, chopin_pitches[c_idx],
            tol_s=tol_s, window_s=sub_window_s,
        )
        f1s.append(r["f1"])
    f1s = np.array(f1s)
    return {
        "median":  float(np.median(f1s)),
        "ci_low":  float(np.percentile(f1s, 2.5)),
        "ci_high": float(np.percentile(f1s, 97.5)),
        "samples": f1s,
    }


def paired_bootstrap_compare(worm_a: np.ndarray, worm_b: np.ndarray,
                              chopin_onsets: np.ndarray,
                              B: int = 1000, window_s: float = 10.0,
                              sub_window_s: float = 5.0, tol_s: float = 0.05,
                              random_state: int = 0) -> dict:
    """ISSUE-021: paired-bootstrap test for H0: F1(a) = F1(b).

    Draws B sub-windows; for each, computes F1(a) and F1(b) on the same
    sub-window; reports the median difference and the (one-sided)
    p-value `P(F1(a) <= F1(b))`.  Useful for asking "is Step 6 *really*
    better than Step 0?" with statistical content.
    """
    from pyannow.targets.midi_target import musical_f1

    rng = np.random.default_rng(random_state)
    diffs = []
    for _ in range(B):
        t0 = rng.uniform(0.0, max(0.0, window_s - sub_window_s))
        t1 = t0 + sub_window_s
        c = chopin_onsets[(chopin_onsets >= t0) & (chopin_onsets <= t1)] - t0
        if len(c) == 0:
            continue
        a = worm_a[(worm_a >= t0) & (worm_a <= t1)] - t0
        b = worm_b[(worm_b >= t0) & (worm_b <= t1)] - t0
        fa = musical_f1(a, c, tol_s=tol_s, window_s=sub_window_s)["f1"] if len(a) else 0.0
        fb = musical_f1(b, c, tol_s=tol_s, window_s=sub_window_s)["f1"] if len(b) else 0.0
        diffs.append(fa - fb)
    diffs = np.array(diffs)
    return {
        "median_diff": float(np.median(diffs)),
        "p_one_sided": float((diffs <= 0).mean()),
        "ci_low":      float(np.percentile(diffs, 2.5)),
        "ci_high":     float(np.percentile(diffs, 97.5)),
    }
