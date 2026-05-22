"""Step 3 — Ridge-regression composer.

NAML material:  Lecture 07 (Least Squares — normal equations, pseudoinverse)
                Lecture 11 (Polynomial regression, ridge regularisation)
                Lab 03     (LS/ridge/kernel regression + PageRank)
                Lab 07     (California housing regression → here: note regression)

We learn a regularised *linear* mapping from the worm's neural latent codes
to Chopin's musical features:

    W_composer ∈ ℝ^{k_worm × k_chopin}   such that   Z_worm @ W_composer ≈ C_chopin

The normal equations give the unregularised solution:

    W* = (Z^T Z)^{-1} Z^T C   (pseudoinverse when Z is full-rank)

Ridge adds a Tikhonov regulariser to handle the ill-conditioned case
(many timesteps, correlated neural features):

    W_ridge = (Z^T Z + λ I)^{-1} Z^T C

NAML insight (L07/L11):  ridge shrinks the singular values of Z by σ_i → σ_i²/(σ_i²+λ).
When σ_i is small (near-degenerate neural direction), ridge truncates it gracefully —
exactly like the pseudoinverse truncates zero singular values.

Lab07 connection:
    In Lab07 we used Ridge to predict California housing prices from features.
    Here the "features" are the worm's neural PCA scores and the "target" is
    the Chopin musical feature matrix.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class RidgeComposer:
    """Regularised linear composer: worm neural latent → musical features.

    Parameters
    ----------
    alpha : regularisation strength λ (chosen by CV if None)
    """

    def __init__(self, alpha: float | None = None):
        self.alpha   = alpha
        self.model   = None
        self.alpha_  = None

    def fit(self, Z_worm: np.ndarray, C_chopin: np.ndarray,
            cv_alphas: list | None = None) -> "RidgeComposer":
        """Fit ridge regression W such that Z_worm @ W ≈ C_chopin.

        Parameters
        ----------
        Z_worm   : (T, k_worm)   worm neural scores (from SVD encoder)
        C_chopin : (T, k_chopin) Chopin musical features (from Procrustes target)
        """
        if self.alpha is None:
            # Cross-validated alpha selection (Lab07 pattern)
            alphas = cv_alphas or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            self.model = RidgeCV(alphas=alphas, cv=5, fit_intercept=True)
        else:
            self.model = Ridge(alpha=self.alpha, fit_intercept=True)

        self.model.fit(Z_worm, C_chopin)
        self.alpha_ = getattr(self.model, "alpha_", self.alpha)
        return self

    def predict(self, Z_worm: np.ndarray) -> np.ndarray:
        """Predict musical features from worm neural scores."""
        return self.model.predict(Z_worm)

    def evaluate(self, Z_test: np.ndarray, C_test: np.ndarray) -> dict:
        """Report MSE, R² and relative error."""
        C_pred = self.predict(Z_test)
        mse    = mean_squared_error(C_test, C_pred)
        ss_tot = np.sum((C_test - C_test.mean(axis=0))**2)
        ss_res = np.sum((C_test - C_pred)**2)
        r2     = 1.0 - ss_res / (ss_tot + 1e-12)
        rel    = np.linalg.norm(C_test - C_pred, "fro") / (np.linalg.norm(C_test, "fro") + 1e-12)
        return {"mse": float(mse), "r2": float(r2), "rel_error": float(rel),
                "alpha": float(self.alpha_ or 0)}

    # ── NAML insight: ridge via SVD (L07) ────────────────────────────────────

    def ridge_via_svd(self, Z: np.ndarray, C: np.ndarray,
                       alpha: float = 1.0) -> np.ndarray:
        """Closed-form ridge solution using the SVD of Z (NAML L07).

        W_ridge = V diag(σ_i/(σ_i²+λ)) U^T C

        This shows exactly how ridge *shrinks* each singular direction:
        directions with small σ_i (noisy neural modes) are suppressed.
        The pseudoinverse is the limit λ → 0 (but singular directions blow up).
        """
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        # Shrinkage factors: σ/(σ²+λ) instead of 1/σ for pseudoinverse
        factors = s / (s**2 + alpha)
        return Vt.T @ np.diag(factors) @ U.T @ C


def explained_variance_by_ridge(Z: np.ndarray, C: np.ndarray,
                                  alphas: list | None = None) -> dict:
    """Show how R² varies with regularisation strength (Lab07 pattern)."""
    alphas  = alphas or np.logspace(-3, 4, 30).tolist()
    Z_tr, Z_te, C_tr, C_te = train_test_split(Z, C, test_size=0.2, random_state=0)
    r2s = []
    for a in alphas:
        m = Ridge(alpha=a).fit(Z_tr, C_tr)
        C_p  = m.predict(Z_te)
        ss_t = np.sum((C_te - C_te.mean(axis=0))**2) + 1e-12
        ss_r = np.sum((C_te - C_p)**2)
        r2s.append(float(1.0 - ss_r / ss_t))
    best_a = alphas[int(np.argmax(r2s))]
    return {"alphas": list(alphas), "r2s": r2s, "best_alpha": best_a}
