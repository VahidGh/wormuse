## ISSUE-026 — Add Lab V diagnostics + Lasso to Step 3 Ridge `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P2 |
| **Severity** | Statistical validity — Ridge fit is reported with no diagnostics |

**Description.** Step 3 fits `RidgeCV` and reports α and R². AppStat Lab V demands the full diagnostic suite on every linear-model fit:

- Residuals vs fitted (linearity check)
- QQ-plot (residual normality)
- **Breusch-Pagan** (heteroskedasticity)
- **Durbin-Watson** (autocorrelation — particularly relevant since the data is a time series!)
- **VIF** (multicollinearity per predictor)
- **Cook's distance** (influential points)

Additionally, Step 3 hardcodes `k_worm = 4` (top-4 PCs). Lab V says: run `LassoCV` and count non-zero coefficients to discover the *true* effective dimensionality the data demands. If Lasso keeps fewer than 4 PCs, the Ridge model is over-specified.

**Fix plan.**

1. Add to `pyannow/step3_regression/ridge_composer.py`:

```python
def diagnose_fit(self, Z, C):
    """Lab V diagnostic table per Chopin feature dim."""
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson
    # ... see wormuse-analytics/src/wormuse_analytics/regression.diagnose_ridge

def lasso_path(self, Z, C, target_col=0):
    """Run LassoCV to discover the true effective k."""
    from sklearn.linear_model import LassoCV
    # ... see wormuse-analytics/src/wormuse_analytics/regression.lasso_path_selection
```

2. New cell in notebook 03 after Step 3's `print(f'Best α (RidgeCV): {rc.alpha_:.4f}')`:

```python
df_diag = rc.diagnose_fit(Z_worm, C_chopin)
display(df_diag.round(3))
lasso = rc.lasso_path(Z_worm, C_chopin, target_col=0)
print(f'Lasso keeps {lasso["n_kept"]}/{k_worm} PCs (alpha={lasso["alpha"]:.4f})')
```

**Note on the time-series caveat.** Durbin-Watson on Step 3's residuals will almost certainly reject independence — the data is a 0.5 ms-spaced time series of muscle voltages. The correct interpretation: ordinary i.i.d. SEs are invalid; either compute Newey-West HAC SEs (`fit.get_robustcov_results('HAC', maxlags=...)`) or treat the per-step F1 + CV as the inferential output, not the OLS SEs.

**Affected files.**
- `PyANNOW/src/pyannow/step3_regression/ridge_composer.py`.
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — new cell after Step 3.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/TODO.md` — this entry.
