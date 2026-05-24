"""Dimensionality reduction — AppStat 2026 Lectures 01-02 / Lab II.

Lecture recap
-------------
**Lecture 01 — PCA (linear).**  StandardScaler then PCA; pick k by
cumulative explained variance >= 0.9 or by the scree-plot elbow; the
**biplot** (scores + loading arrows together) is what makes PCA
interpretable.

**Lecture 02 — Nonlinear.**  `TSNE(perplexity=30, init='pca')` for local
structure; `UMAP(n_neighbors=15, min_dist=0.1)` for a faster nonlinear
view that preserves more global geometry.  Caveats: in t-SNE, inter-cluster
distance and cluster size are *not* meaningful.

For PyANNOW the audit findings are:
- PCA: ✅ done (Step 1a RSVD, Step 2 PCA) but ❌ no biplot.
- t-SNE/UMAP: ❌ entirely missing — issued as ISSUE-024.
"""
from __future__ import annotations

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_with_scree(X: np.ndarray, standardize: bool = True, var_threshold: float = 0.90):
    """Lab II: StandardScaler -> PCA, return (Z_scores, pca_obj, k_chosen).

    `k_chosen` is the smallest number of components reaching var_threshold.
    """
    if standardize:
        Xs = StandardScaler().fit_transform(X)
    else:
        Xs = X - X.mean(axis=0, keepdims=True)
    pca = PCA()
    Z = pca.fit_transform(Xs)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, var_threshold) + 1)
    return Z, pca, k


def biplot(Z: np.ndarray, components: np.ndarray, ax=None,
           feature_names: list[str] | None = None,
           sample_colors=None, arrow_scale: float = 5.0,
           max_arrows: int = 15):
    """Lab II biplot — scores (scatter) + top loadings (arrows).

    Parameters
    ----------
    Z : (n, p)               PCA scores from .fit_transform
    components : (p, d)      pca.components_  — loadings on each feature
    feature_names : list[d]  names of the original features (for arrow labels)
    arrow_scale : scale on the loadings for visibility
    max_arrows : show only the top loadings by |L2| length (keep plot readable)

    AppStat caveat: SVD sign is ambiguous; the absolute direction of an
    arrow is unique up to a global flip.  Interpret signs cautiously.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.4, c=sample_colors)
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set(xlabel="PC1", ylabel="PC2", title="Lab II biplot — scores + loadings")

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(components.shape[1])]

    norms = np.sqrt(components[0]**2 + components[1]**2)
    top = np.argsort(-norms)[:max_arrows]
    for i in top:
        ax.arrow(0, 0,
                 components[0, i] * arrow_scale,
                 components[1, i] * arrow_scale,
                 head_width=0.05, color="red", alpha=0.7)
        ax.text(components[0, i] * arrow_scale * 1.1,
                components[1, i] * arrow_scale * 1.1,
                str(feature_names[i]), color="red", fontsize=8)
    return ax


def nonlinear_view(X: np.ndarray, method: str = "umap",
                   n_components: int = 2, random_state: int = 0,
                   standardize: bool = True, **kwargs):
    """Lab II nonlinear projection.  method in {'umap', 'tsne'}.

    Standardizes (recommended) before projection.  Passes through `**kwargs`
    to the underlying estimator: `perplexity` for t-SNE, `n_neighbors` /
    `min_dist` for UMAP.
    """
    if standardize:
        Xs = StandardScaler().fit_transform(X)
    else:
        Xs = X

    if method == "tsne":
        from sklearn.manifold import TSNE
        defaults = dict(perplexity=30, init="pca", learning_rate="auto",
                        random_state=random_state)
        defaults.update(kwargs)
        return TSNE(n_components=n_components, **defaults).fit_transform(Xs)
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "pip install umap-learn — required for the UMAP view") from exc
        defaults = dict(n_neighbors=15, min_dist=0.1, random_state=random_state)
        defaults.update(kwargs)
        return umap.UMAP(n_components=n_components, **defaults).fit_transform(Xs)
    raise ValueError(f"unknown method {method!r}; use 'umap' or 'tsne'")
