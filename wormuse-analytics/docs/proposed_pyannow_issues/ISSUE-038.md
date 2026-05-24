## ISSUE-038 — No cross-validation anywhere `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | ✅ Resolved — v0.8.0 |
| **Priority** | P2 |
| **Severity** | Statistical validity — every per-step F1 is an in-sample point estimate |

**Description.** Every NAML step (1b Procrustes, 3 Ridge, 4-6 MLP) was fitted and evaluated on the same 10s Chopin window. Every reported F1 was in-sample — systematically overestimating generalisation performance. Random splits are wrong here because residuals are autocorrelated (Durbin-Watson failure, ISSUE-026).

**Fix (v0.8.0).** New module `pyannow/training/cv.py`:

```python
from pyannow.training.cv import time_series_cv, blocked_bootstrap_ci

# Time-series K-fold CV (no future leakage)
cv = time_series_cv(
    fit_fn   = lambda Z, C: RidgeComposer(alpha=None).fit(Z, C),
    score_fn = lambda m, Z, C: musical_f1(m.predict(Z), onsets(C))["f1"],
    Z=Z_worm, C=C_chopin, n_splits=5,
)
print(f"Step 3 CV F1: {cv['mean']:.3f} ± {cv['std']:.3f}")

# Block bootstrap CI (preserves autocorrelation)
bb = blocked_bootstrap_ci(fit_fn, score_fn, Z_worm, C_chopin, block_len=20, n_boot=200)
print(f"Step 3 bootstrap 95% CI: [{bb['ci_low']:.3f}, {bb['ci_high']:.3f}]")
```

**Key implementation details:**
- `time_series_cv()`: Walk-forward K-fold — test block is always in the future relative to train set. No future leakage. `TimeSeriesSplit` semantics (train always precedes test in time).
- `blocked_bootstrap_ci()`: Block-bootstrap resamples contiguous blocks to preserve autocorrelation structure (ISSUE-026 / Durbin-Watson fix).
- Both functions: generic `fit_fn(Z, C) → model` and `score_fn(model, Z, C) → float` API — works with any step.

**Tested.** `test_training_cv.py` — 12 tests covering:
- Expected keys, n_folds ≤ n_splits, no future leakage, determinism, ValueError on short data
- Bootstrap CI contains mean, seed reproducibility

**AppStat connection.** L06: cross-validation is the standard approach for unbiased model evaluation. `TimeSeriesSplit` (consecutive blocks) is mandatory when observations are autocorrelated.

**Note on Durbin-Watson (ISSUE-026).** Block bootstrap addresses the autocorrelation problem for CIs. For full OLS diagnostic on Ridge residuals, see ISSUE-026 (still open — notebook cell needed).

**Category:** `Category C — Statistical Validation`
