"""Classification — AppStat 2026 Lectures 05-06 / Lab VI.

Lecture recap
-------------
**Lecture 05 — Logistic regression.**  Models `P(y=1|x) = sigmoid(x^T beta)`.
Coefficients are log-odds.  `LogisticRegression(C=...)` for prediction;
`sm.Logit` for inference (p-values, CIs).

**Lecture 06 — Classification metrics + model selection.**  Confusion
matrix; precision/recall/F1; ROC (TPR vs FPR + AUC); PR curve (precision
vs recall — preferred on heavy imbalance); stratified k-fold CV;
GridSearchCV wrapped in Pipeline to prevent leakage; class-imbalance
handling (`class_weight='balanced'`, threshold tuning).  Choose the
operating threshold:

- Default `predict()` uses 0.5 — rarely right.
- **Youden's J**: maximise `TPR - FPR` (balanced).
- **Best F1**:  maximise F1 over a threshold sweep.
- Application-specific cost: set recall floor, report precision.

For PyANNOW the audit findings:
- F1 reported at one tolerance ✅ (ISSUE-016) but ❌ no curves, no CIs,
  no CV — covered by ISSUE-020 / ISSUE-021.
- The hardcoded `find_peaks(height=mean(activ))` threshold across every
  step is a 1-feature classifier with a magic constant — covered by
  ISSUE-027.

This module implements both.
"""
from __future__ import annotations

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                              roc_auc_score, roc_curve)


def _onsets_to_bin_labels(onsets: np.ndarray, t_arr: np.ndarray,
                          tol_s: float = 0.05) -> np.ndarray:
    """Convert onset times to a per-bin binary label aligned with t_arr.

    A bin at time t is labelled 1 if any target onset falls within tol_s of t.
    This is the AppStat-correct binary-classification framing of the
    onset-detection problem.
    """
    y = np.zeros_like(t_arr, dtype=int)
    for o in onsets:
        mask = np.abs(t_arr - o) <= tol_s
        y[mask] = 1
    return y


def precision_recall_curve_onsets(activ: np.ndarray, chopin_onsets: np.ndarray,
                                   t_arr: np.ndarray, tol_s: float = 0.05):
    """Lab VI PR curve treating the activation envelope as the soft predictor.

    Returns (precision, recall, thresholds, auc_pr).
    """
    y = _onsets_to_bin_labels(chopin_onsets, t_arr, tol_s=tol_s)
    p, r, thr = precision_recall_curve(y, activ)
    ap = average_precision_score(y, activ)
    return p, r, thr, float(ap)


def roc_curve_onsets(activ: np.ndarray, chopin_onsets: np.ndarray,
                     t_arr: np.ndarray, tol_s: float = 0.05):
    """Lab VI ROC curve.  Returns (fpr, tpr, thresholds, auc_roc)."""
    y = _onsets_to_bin_labels(chopin_onsets, t_arr, tol_s=tol_s)
    fpr, tpr, thr = roc_curve(y, activ)
    return fpr, tpr, thr, float(roc_auc_score(y, activ))


def logistic_onset_detector(activ: np.ndarray, chopin_onsets: np.ndarray,
                             t_arr: np.ndarray, tol_s: float = 0.05,
                             threshold_strategy: str = "youden") -> dict:
    """Fit a logistic regression on the activation to detect onsets.

    The "activation" is the smoothed envelope produced by a NAML step
    (Step 1b's |Z_aligned|.max(axis=1), Step 3's |C_pred_ridge|.max,
    or any other per-timestep magnitude).  We fit a single-feature
    logistic model `P(onset|activ_t)` and pick the operating threshold
    by `threshold_strategy in {'youden', 'best_f1', 'default'}`.

    Returns a dict with: clf, threshold, f1, precision, recall, auc_roc.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    y = _onsets_to_bin_labels(chopin_onsets, t_arr, tol_s=tol_s)
    X = activ.reshape(-1, 1)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X, y)
    prob = clf.predict_proba(X)[:, 1]

    if threshold_strategy == "default":
        thr = 0.5
    elif threshold_strategy == "youden":
        fpr, tpr, t = roc_curve(y, prob)
        j = tpr - fpr
        thr = float(t[int(np.argmax(j))])
    elif threshold_strategy == "best_f1":
        p, r, t = precision_recall_curve(y, prob)
        f1_curve = 2 * p * r / np.where((p + r) > 0, (p + r), 1.0)
        thr = float(t[int(np.argmax(f1_curve[:-1]))]) if len(t) > 0 else 0.5
    else:
        raise ValueError(f"unknown strategy: {threshold_strategy!r}")

    y_pred = (prob >= thr).astype(int)
    return {
        "clf":       clf,
        "threshold": thr,
        "f1":        float(f1_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall":    float(recall_score(y, y_pred, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y, prob)),
    }
