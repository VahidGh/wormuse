## ISSUE-038 — No cross-validation anywhere `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P2 |
| **Severity** | Statistical validity — every per-step F1 is an in-sample point estimate |

**Description.** PyANNOW notebook 03 reports F1 / onset_loss per step computed on the same 10 s Chopin window used to *fit* the model (Step 1b's Procrustes, Step 3's RidgeCV, Steps 4-6's MLP, all train on `(Z_worm, C_chopin)`). There is no held-out split, no temporal cross-validation, no estimate of variability.

Result: every score is **in-sample**. The published per-step F1 systematically overestimates what the worm would achieve on a Chopin window the model hasn't seen.

AppStat Lecture 06 minimum:

- **Time-series CV** — split the 10 s window into K folds (e.g. K=5: fit on 8 s, test on the held-out 2 s; rotate). Report mean + std over folds.
- **Bootstrap CIs** — already covered by ISSUE-021 for the *test* metric, but here we add CV for the *fit*.
- **No leakage** — the calibrated threshold (ISSUE-027 / 033) must be tuned on the train fold, evaluated on the test fold.

**Fix plan.** Add `pyannow/training/cv.py`:

```python
def time_series_cv(fit_fn, score_fn, Z, C, n_splits=5, window_s=10.0) -> dict:
    """Time-series K-fold cross-validation.

    For k in 0..K-1:
      train  = timesteps in [0, window_s) excluding the k-th interval
      test   = the k-th interval
      fit    = fit_fn(Z[train], C[train])
      pred   = fit.predict(Z[test])
      score  = score_fn(pred, C[test])
    Returns mean and std of score across folds.
    """
    ...
```

In notebook 03, wrap each step's fit + score in this:

```python
cv = time_series_cv(
    fit_fn=lambda Z, C: RidgeComposer(alpha=None).fit(Z, C),
    score_fn=lambda pred, C_true: musical_f1_from_features(pred, C_true, ...),
    Z=Z_worm, C=C_chopin, n_splits=5, window_s=DURATION,
)
print(f'Step 3 CV F1: {cv["mean"]:.3f} ± {cv["std"]:.3f}')
```

**Note on Durbin-Watson.** Time-series CV is also the right answer to the Durbin-Watson failure flagged in ISSUE-026. Independent random splits (StratifiedKFold) are *wrong* here because the residuals are autocorrelated; consecutive timesteps belong together. Use `TimeSeriesSplit` from `sklearn.model_selection`.

**Affected files.**
- `PyANNOW/src/pyannow/training/cv.py` — new module.
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — wrap each step in `time_series_cv`.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/tests/test_training.py` — reproducibility test with seeded folds.
- `PyANNOW/TODO.md` — this entry.
