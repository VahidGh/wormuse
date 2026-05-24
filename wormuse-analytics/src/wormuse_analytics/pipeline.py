"""Improved worm→Chopin pipeline — the goal-driven re-engineering.

This module wires the AppStat-corrected version of the 8-step PyANNOW pipeline.
It does NOT replace PyANNOW; it re-uses every component (forward model, RSVD,
PCA, Ridge, MLP) but inserts proper statistical machinery where PyANNOW skipped
it.

Recipe (closes the 10 logic problems documented in
docs/STATISTICAL_DIAGNOSTICS.md §2):

    1. Drop synthetic 302-neuron matrix; PCA directly on (T, 95) muscles.
    2. StandardScaler before any linear method.
    3. Predict pitch (not just onset) by inverse-PCA back to piano-roll.
    4. Replace `find_peaks(height=mean)` with calibrated LogisticRegression
       + Youden's-J threshold tuning.
    5. Cross-validate via temporal sub-windows.
    6. Score with the composite (pitch_aware_f1, AUC-PR, IOI similarity,
       velocity correlation) + bootstrap 95% CIs.

The module is thin — it composes the per-lecture submodules
(descriptive / dimreduction / clustering / regression / classification /
trees / metrics) into a runnable pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from . import descriptive, dimreduction, regression, classification, trees, metrics


@dataclass
class ImprovedRun:
    """One row of the improved pipeline's output."""
    step:                  str
    n_notes:               int
    pitch_aware_f1:        float
    auc_pr:                float
    ioi_similarity:        float
    velocity_correlation:  float
    bootstrap_ci_low:      float
    bootstrap_ci_high:     float


@dataclass
class ImprovedPipeline:
    """The AppStat-corrected pipeline.

    Input:  the cached PyANNOW outputs (see loaders.run_pyannow_pipeline()).
    Output: a DataFrame with one row per step, columns =
            (n_notes, pitch_aware_f1, auc_pr, ioi_sim, vel_corr, ci_low, ci_high).

    The pipeline is purposefully **deterministic** given the seed in the cache.
    """
    duration_s:   float
    chopin_onsets: np.ndarray
    chopin_pitches: np.ndarray
    chopin_velocities: np.ndarray
    t_arr:         np.ndarray
    seed:          int = 0

    def score_step(self, name: str, onsets: np.ndarray,
                   pitches: np.ndarray, velocities: np.ndarray,
                   activation: Optional[np.ndarray] = None) -> ImprovedRun:
        """Score one step's output against Chopin using the composite metric."""
        # Pitch-aware F1 with bipartite matching (see metrics.pitch_aware_f1)
        pf1 = metrics.pitch_aware_f1(
            onsets, pitches,
            self.chopin_onsets, self.chopin_pitches,
            tol_s=0.05, window_s=self.duration_s,
        )["f1"]

        # AUC-PR — only defined when we have a continuous activation envelope
        if activation is not None:
            _, _, _, auc_pr = classification.precision_recall_curve_onsets(
                activation, self.chopin_onsets, self.t_arr, tol_s=0.05)
        else:
            auc_pr = float("nan")

        # IOI similarity — wraps the existing pyannow function
        from pyannow.targets.midi_target import ioi_similarity
        ioi = ioi_similarity(onsets, self.chopin_onsets, window_s=self.duration_s)

        # Velocity correlation — Pearson r over time-matched notes
        vel_r = metrics.velocity_correlation(
            onsets, velocities,
            self.chopin_onsets, self.chopin_velocities,
            tol_s=0.05,
        )

        # Bootstrap 95% CI on pitch-aware F1
        ci = metrics.bootstrap_pitch_aware_f1(
            onsets, pitches,
            self.chopin_onsets, self.chopin_pitches,
            B=400, window_s=self.duration_s, sub_window_s=5.0,
            tol_s=0.05, random_state=self.seed,
        )
        return ImprovedRun(
            step=name, n_notes=int(len(onsets)),
            pitch_aware_f1=float(pf1), auc_pr=float(auc_pr),
            ioi_similarity=float(ioi), velocity_correlation=float(vel_r),
            bootstrap_ci_low=float(ci["ci_low"]),
            bootstrap_ci_high=float(ci["ci_high"]),
        )

    def score_all(self, runs: list[tuple]) -> pd.DataFrame:
        """Score a list of (name, onsets, pitches, velocities, activation) tuples."""
        return pd.DataFrame([
            self.score_step(*r).__dict__ for r in runs
        ])


def reachable_pitches(muscle_pitches: np.ndarray,
                      chopin_pitches: np.ndarray) -> dict:
    """Closes logic problem #3 (pitch bottleneck).

    Reports the fraction of Chopin pitches reachable by *any* of the
    n_muscles pitches in muscle_pitches.  A reachable pitch is one where
    pitch_class(chopin_p) == pitch_class(muscle_p) for some muscle_p.

    Returns:
        reachable_fraction : in [0, 1]
        n_reachable        : count of Chopin pitches reachable
        ceiling_f1         : upper bound on F1 achievable under this map
                             (recall ceiling; precision is whatever the model gives)
    """
    chopin_classes = set(int(p) % 12 for p in chopin_pitches)
    muscle_classes = set(int(p) % 12 for p in muscle_pitches)
    reachable_classes = chopin_classes & muscle_classes

    n_reach = int(sum(1 for p in chopin_pitches if (int(p) % 12) in reachable_classes))
    frac = n_reach / max(1, len(chopin_pitches))

    # F1 ceiling under perfect timing and matching pitch-class: recall = frac,
    # precision = 1.0, so F1_max = 2 * frac / (frac + 1).
    f1_ceiling = 2.0 * frac / (frac + 1.0) if frac > 0 else 0.0
    return {
        "n_chopin_pitches":        len(chopin_pitches),
        "n_reachable":             n_reach,
        "reachable_fraction":      float(frac),
        "ceiling_pitch_aware_f1":  float(f1_ceiling),
        "chopin_pitch_classes":    sorted(chopin_classes),
        "muscle_pitch_classes":    sorted(muscle_classes),
        "reachable_classes":       sorted(reachable_classes),
    }


def diagnose_synthetic_neural(X_neural: np.ndarray,
                              n_muscles_assumed: int = 8) -> dict:
    """Closes logic problem #1 (the 302-neuron matrix is fake).

    Demonstrates that the rank of X_neural is bounded by n_muscles_assumed
    (modulo noise) — runs SVD and reports the variance captured by the first
    n_muscles_assumed components.  An honest 302-D neural manifold would have
    spread variance well beyond 8 components.

    Returns:
        cumvar_at_n_muscles : cumulative variance ratio at component n_muscles_assumed
        effective_rank      : smallest k with cumvar >= 0.999
        gini_concentration  : Gini coefficient of the variance distribution
                              (1.0 = all variance in one component)
    """
    Xc = X_neural - X_neural.mean(axis=1, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    var = sv ** 2 / (sv ** 2).sum()
    cumvar = np.cumsum(var)
    eff_rank = int(np.searchsorted(cumvar, 0.999) + 1)

    sorted_var = np.sort(var)[::-1]
    n = len(sorted_var)
    gini = float(1.0 - 2.0 * np.sum((np.arange(1, n+1) - 0.5) * sorted_var) / (n * sorted_var.sum()))

    return {
        "shape":                     X_neural.shape,
        "n_muscles_assumed":         n_muscles_assumed,
        "cumvar_at_n_muscles":       float(cumvar[min(n_muscles_assumed-1, n-1)]),
        "effective_rank_999":        eff_rank,
        "gini_variance_concentration": gini,
        "top_5_variance_ratios":     [float(v) for v in var[:5]],
        "interpretation": ("If cumvar_at_n_muscles >= 0.99, the matrix is "
                            "synthetic muscle replication, not a 302-D manifold. "
                            "If effective_rank << 302, the same conclusion holds."),
    }
