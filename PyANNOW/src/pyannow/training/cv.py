"""Time-series cross-validation for PyANNOW pipeline steps.

AppStat 2026 material: Lecture 06 (cross-validation, bootstrap)
                       Note on Durbin-Watson: ISSUE-026 / ISSUE-038

Why time-series CV?
-------------------
All NAML steps (1b Procrustes, 3 Ridge, 4-6 MLP/Adam/L-BFGS) are fitted and
evaluated on the **same** 10-second Chopin window — producing in-sample F1
scores that overestimate generalisation performance.  Independent random splits
(StratifiedKFold) are incorrect here because the residuals are autocorrelated
(Durbin-Watson test fails, ISSUE-026): consecutive timesteps are not i.i.d.

Solution: ``TimeSeriesSplit``-style K-fold where each fold uses a contiguous
test block and trains on everything before it.  This respects temporal order
and gives an unbiased estimate of out-of-sample performance.

Usage::

    from pyannow.training.cv import time_series_cv

    cv = time_series_cv(
        fit_fn   = lambda Z, C: RidgeComposer(alpha=None).fit(Z, C),
        score_fn = lambda fit, Z, C: musical_f1(fit.predict(Z), onsets(C))["f1"],
        Z        = Z_worm,
        C        = C_chopin,
        n_splits = 5,
    )
    print(f"CV F1: {cv['mean']:.3f} ± {cv['std']:.3f}")
"""
from __future__ import annotations

import numpy as np


def time_series_cv(
    fit_fn,
    score_fn,
    Z:        np.ndarray,
    C:        np.ndarray,
    n_splits: int = 5,
) -> dict:
    """Time-series K-fold cross-validation.

    Splits the T timesteps of (Z, C) into ``n_splits`` contiguous blocks.
    For fold k:
        train = all blocks except block k  (indices concat)
        test  = block k
        model = fit_fn(Z[train], C[train])
        score = score_fn(model, Z[test], C[test])

    The test blocks are chosen as the **last** n_splits equal-width segments
    and the train set always precedes the test set in time (no future leakage).
    Concretely, for K=5 and T=200:
        fold 0: train [0:160],  test [160:200]
        fold 1: train [0:120],  test [120:160]
        fold 2: train [0:80],   test [80:120]
        fold 3: train [0:40],   test [40:80]
        fold 4: train []         → skipped (too little training data)

    Only folds with ≥ n_splits timesteps in the train set are evaluated.

    Parameters
    ----------
    fit_fn   : callable(Z_train, C_train) → fitted model object
    score_fn : callable(model, Z_test, C_test) → float score
    Z        : (T, d_Z) worm neural feature matrix
    C        : (T, d_C) Chopin feature matrix
    n_splits : number of folds (default 5)

    Returns
    -------
    dict:
        mean      : mean score across folds
        std       : std of scores across folds
        scores    : list of per-fold scores
        n_folds   : number of evaluated folds
        fold_info : list of dicts with train_size, test_size, test_start, test_end
    """
    T = len(Z)
    if T < n_splits * 2:
        raise ValueError(
            f"T={T} is too short for n_splits={n_splits}: need at least {n_splits * 2} timesteps."
        )

    block = T // n_splits
    scores = []
    fold_info = []

    # Walk backwards: last block is fold 0, second-last is fold 1, ...
    for k in range(n_splits):
        test_end   = T - k * block
        test_start = test_end - block
        if test_start <= n_splits:
            # Too little training data — skip this fold
            break

        train_idx = np.arange(0, test_start)
        test_idx  = np.arange(test_start, test_end)

        Z_tr, C_tr = Z[train_idx], C[train_idx]
        Z_te, C_te = Z[test_idx],  C[test_idx]

        model = fit_fn(Z_tr, C_tr)
        sc    = score_fn(model, Z_te, C_te)
        scores.append(float(sc))
        fold_info.append({
            "fold":       k,
            "train_size": len(train_idx),
            "test_size":  len(test_idx),
            "test_start": int(test_start),
            "test_end":   int(test_end),
            "score":      float(sc),
        })

    if not scores:
        return {"mean": 0.0, "std": 0.0, "scores": [], "n_folds": 0, "fold_info": []}

    arr = np.array(scores)
    return {
        "mean":      float(arr.mean()),
        "std":       float(arr.std()),
        "scores":    scores,
        "n_folds":   len(scores),
        "fold_info": fold_info,
    }


def blocked_bootstrap_ci(
    fit_fn,
    score_fn,
    Z:         np.ndarray,
    C:         np.ndarray,
    block_len: int   = 20,
    n_boot:    int   = 200,
    ci:        float = 0.95,
    seed:      int   = 42,
) -> dict:
    """Block-bootstrap confidence interval for time-series scores.

    Samples blocks of length ``block_len`` with replacement to form bootstrap
    replicates of (Z, C).  Each replicate is refit and scored.  Block
    bootstrap preserves local autocorrelation structure (unlike i.i.d.
    bootstrap which breaks temporal dependencies).

    Parameters
    ----------
    block_len : length of each bootstrap block (should be ≥ autocorrelation lag)
    n_boot    : number of bootstrap replicates
    ci        : confidence level (default 0.95)

    Returns
    -------
    dict: mean, ci_low, ci_high, std, n_boot
    """
    rng = np.random.default_rng(seed)
    T = len(Z)
    n_blocks = T // block_len
    if n_blocks < 2:
        raise ValueError(f"T={T} too short for block_len={block_len}: need ≥ {2*block_len} timesteps.")

    boot_scores = []
    for _ in range(n_boot):
        # Sample blocks with replacement
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([
            np.arange(b * block_len, min((b + 1) * block_len, T))
            for b in chosen
        ])
        Z_b = Z[idx[:T]]
        C_b = C[idx[:T]]
        model = fit_fn(Z_b, C_b)
        sc    = score_fn(model, Z_b, C_b)
        boot_scores.append(float(sc))

    arr   = np.array(boot_scores)
    alpha = (1.0 - ci) / 2.0
    return {
        "mean":   float(arr.mean()),
        "ci_low": float(np.quantile(arr, alpha)),
        "ci_high":float(np.quantile(arr, 1.0 - alpha)),
        "std":    float(arr.std()),
        "n_boot": n_boot,
    }
