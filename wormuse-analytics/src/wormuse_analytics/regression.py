"""Linear models + diagnostics — AppStat 2026 Lecture 04 / Lab V.

Lecture recap
-------------
`statsmodels.api.OLS` is the inference workhorse — it returns coefficients,
standard errors, p-values, R^2, and the F-test.  The Lab V diagnostic
suite:

- **Residuals vs fitted** — linearity (random scatter = OK; pattern = bad).
- **QQ-plot of residuals** — normality of errors.
- **Breusch-Pagan** — heteroskedasticity (p<0.05: residuals are non-constant).
- **Durbin-Watson** — autocorrelation of residuals (≈2 = no autocorrelation).
- **VIF** — multicollinearity per predictor (>5 troublesome, >10 severe).
- **Cook's distance** — influential observations (>4/n flagged).

Regularisation:
- **Ridge** shrinks all coefficients; stabilises ill-conditioned X^T X.
- **Lasso** induces sparsity (does feature selection); coefficients set to 0.
- **Elastic Net** = mix of L1 and L2 — both shrinkage and selection.

Always **standardize features** before regularised regression: the penalty
depends on the scale of the predictors.

For PyANNOW: Ridge ✅ (Step 3 with RidgeCV) but ❌ no diagnostics, ❌ no
Lasso comparison — issued as ISSUE-026.  Notably: although the input
PC scores are mathematically orthogonal (PCA), the diagnostics should
still be reported because the *standardised* Ridge estimator can still
suffer from outlier leverage or non-normal residuals.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


@dataclass
class DiagnosticsReport:
    """Per-target-dim Lab V diagnostics summary."""
    target_dim:     int
    r2:             float
    bp_pvalue:      float
    dw_statistic:   float
    vifs:           np.ndarray
    n_high_vif:     int            # count of VIF > 5
    n_influential:  int            # Cook's distance > 4/n
    shapiro_p:      float | None


def diagnose_ols(X: np.ndarray, y: np.ndarray, var_names=None) -> DiagnosticsReport:
    """Run the full Lab V diagnostic suite for a single target y."""
    from scipy.stats import shapiro

    X1 = sm.add_constant(X)
    fit = sm.OLS(y, X1).fit()
    bp_stat, bp_p, _, _ = het_breuschpagan(fit.resid, fit.model.exog)
    dw = durbin_watson(fit.resid)

    vifs = np.array([variance_inflation_factor(X1, i) for i in range(X1.shape[1])])
    infl = fit.get_influence()
    cooks = infl.cooks_distance[0]
    thresh = 4.0 / len(y)
    n_inf = int((cooks > thresh).sum())

    # Shapiro is unreliable for n > 5000; skip in that case
    if len(fit.resid) <= 5000:
        sp = float(shapiro(fit.resid).pvalue)
    else:
        sp = None

    return DiagnosticsReport(
        target_dim=-1,
        r2=float(fit.rsquared),
        bp_pvalue=float(bp_p),
        dw_statistic=float(dw),
        vifs=vifs,
        n_high_vif=int((vifs > 5.0).sum()),
        n_influential=n_inf,
        shapiro_p=sp,
    )


def diagnose_ridge(Z: np.ndarray, C: np.ndarray,
                   alpha: float | None = None,
                   standardize: bool = True) -> pd.DataFrame:
    """Diagnose the Step 3 Ridge fit: one row per Chopin feature dim.

    Ridge has no closed-form OLS analogue for std errors, so the diagnostics
    are performed on the *equivalent OLS* fit (alpha=0) per target column;
    Ridge's role is only to choose alpha by CV.  This separates the
    statistical-validity question (Lab V) from the regularisation question.
    """
    if standardize:
        Z = StandardScaler().fit_transform(Z)

    rows = []
    for j in range(C.shape[1]):
        rep = diagnose_ols(Z, C[:, j])
        rows.append({
            "target_dim":   j,
            "r2":           rep.r2,
            "bp_pvalue":    rep.bp_pvalue,
            "dw":           rep.dw_statistic,
            "max_vif":      float(rep.vifs.max()),
            "n_high_vif":   rep.n_high_vif,
            "n_influential": rep.n_influential,
            "shapiro_p":    rep.shapiro_p,
        })
    return pd.DataFrame(rows)


def ridge_cv(Z: np.ndarray, C: np.ndarray,
             alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)) -> tuple[RidgeCV, float]:
    """RidgeCV mirroring PyANNOW's Step 3.  Returns (model, best_alpha)."""
    Zs = StandardScaler().fit_transform(Z)
    model = RidgeCV(alphas=list(alphas), cv=5).fit(Zs, C)
    return model, float(model.alpha_)


def lasso_path_selection(Z: np.ndarray, C: np.ndarray, target_col: int = 0):
    """Lab V Lasso for feature selection.

    Fits LassoCV on (Z -> C[:, target_col]) and reports which features
    have non-zero coefficients at the chosen alpha.  Sparsity tells us
    how many PCs the model *actually* uses — independent of what
    `k_worm=4` was set to.
    """
    Zs = StandardScaler().fit_transform(Z)
    model = LassoCV(cv=5, max_iter=10000, random_state=0).fit(Zs, C[:, target_col])
    nonzero = np.where(model.coef_ != 0.0)[0]
    return {
        "alpha":    float(model.alpha_),
        "n_kept":   int(len(nonzero)),
        "kept_idx": nonzero,
        "coef":     model.coef_,
        "r2":       float(model.score(Zs, C[:, target_col])),
    }
