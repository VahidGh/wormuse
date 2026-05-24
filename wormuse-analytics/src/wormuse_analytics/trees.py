"""Tree-based methods — AppStat 2026 Lecture 07.

Lecture recap
-------------
**Decision trees** partition the feature space by axis-aligned splits
(Gini / entropy for classification, MSE for regression).  High variance
on their own — one tree is too noisy.

**Random forests** = bagging + random feature subsets per node.  Average
many decorrelated trees, dramatically reducing variance.  Free validation
via **out-of-bag** error.  Feature importance: MDI (biased toward
high-cardinality features) vs permutation importance (more reliable).

**Gradient boosting** sequentially fits each new tree to the residuals
of the cumulative model — gradient descent in function space.  Higher
accuracy than RF on tabular tasks, but slower and more sensitive to
hyperparameters.

For PyANNOW: ❌ no trees anywhere.  The NAML pipeline goes
Linear (Ridge) → MLP (Adam) → L-BFGS → PINN.  AppStat would insist on
fitting a Random Forest *first* and only escalating to deep models if
the RF leaves significant performance on the table.  This module
implements the RF baseline as a **model-agnostic ceiling** — issued
under ISSUE-028.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler


@dataclass
class RFBaselineReport:
    """Random Forest regression report for the worm-to-Chopin task."""
    r2_train:           float
    r2_oob:             float
    feature_importance: np.ndarray  # MDI importance per PC
    permutation_means:  np.ndarray  # permutation importance mean
    permutation_stds:   np.ndarray  # permutation importance std


def rf_baseline(Z: np.ndarray, C: np.ndarray,
                 n_estimators: int = 500, random_state: int = 0,
                 max_depth: int | None = None) -> tuple[RandomForestRegressor, RFBaselineReport]:
    """Lab VII RF: regress Chopin features C on neural PC scores Z.

    Z is the (T, k_worm) PCA / SVD scores from Step 1/2; C is the
    (T, k_chopin) Chopin feature matrix.  Returns the fitted model plus
    a structured report of OOB R^2 and feature importance.
    """
    Zs = StandardScaler().fit_transform(Z)
    rf = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=2, n_jobs=-1, oob_score=True,
        random_state=random_state).fit(Zs, C)
    r2_train = float(rf.score(Zs, C))
    r2_oob   = float(rf.oob_score_)

    pi = permutation_importance(rf, Zs, C, n_repeats=10,
                                random_state=random_state, n_jobs=-1)
    return rf, RFBaselineReport(
        r2_train=r2_train, r2_oob=r2_oob,
        feature_importance=rf.feature_importances_,
        permutation_means=pi.importances_mean,
        permutation_stds=pi.importances_std,
    )


def rf_predicted_onsets(rf: RandomForestRegressor, Z: np.ndarray, t_arr: np.ndarray,
                         min_interval_s: float = 0.28) -> tuple[np.ndarray, np.ndarray]:
    """Convert RF predictions to (onsets, activation_envelope) like a NAML step.

    Mirrors the pattern used in PyANNOW: take the max-magnitude column of
    the predicted feature matrix as the activation envelope, then run
    `find_peaks` with the BWM refractory distance.
    """
    from scipy.signal import find_peaks
    Zs = StandardScaler().fit_transform(Z)
    C_pred = rf.predict(Zs)
    activ  = np.abs(C_pred).max(axis=1)
    dt     = float(t_arr[1] - t_arr[0])
    distance_samples = max(1, int(min_interval_s / dt))
    peaks, _ = find_peaks(activ, distance=distance_samples, height=activ.mean())
    return t_arr[peaks], activ
