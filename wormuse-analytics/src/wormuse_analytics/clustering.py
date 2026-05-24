"""Clustering — AppStat 2026 Lecture 03 / Labs III + IV.

Lecture recap
-------------
Four clustering paradigms:

1. **KMeans** — spherical clusters, fixed k.  Choose k by silhouette / elbow.
2. **Agglomerative (hierarchical)** — `linkage` (Ward / single / complete / average)
   + `dendrogram` + `cophenet` correlation as a quality check.
3. **DBSCAN** — density-based; finds arbitrary shapes, labels noise as -1.
   Choose `eps` by k-distance elbow.
4. **GMM** — probabilistic with soft assignments and elliptical clusters.
   Choose k by BIC.

Adjusted Rand Index (ARI) compares two labelings (1 = perfect agreement,
0 = random).  Silhouette score evaluates a single clustering on the data
( >0.7 strong, 0.5-0.7 reasonable, 0.25-0.5 weak, <0.25 no structure ).

For PyANNOW: KMeans ✅ (Step 2), Ward / DBSCAN / GMM ❌ — issued as ISSUE-025.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score


def kmeans_labels(X: np.ndarray, k: int, random_state: int = 0) -> np.ndarray:
    """Lab III KMeans with k-means++ init and n_init=10."""
    return KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X)


def ward_labels_and_dendrogram(X: np.ndarray, k: int, ax=None):
    """Lab III Ward hierarchical clustering.  Returns (labels, linkage_Z, cophenet_c).

    Cuts the dendrogram at `maxclust=k`.  Also reports the cophenetic
    correlation between the linkage and the original pairwise distance
    matrix — values close to 1 mean the dendrogram preserves distances
    faithfully.
    """
    from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
    from scipy.spatial.distance import pdist

    D = pdist(X, metric="euclidean")
    Z = linkage(D, method="ward")
    c, _ = cophenet(Z, D)
    labels = fcluster(Z, t=k, criterion="maxclust")
    if ax is not None:
        dendrogram(Z, color_threshold=Z[-(k-1), 2] if k > 1 else 0, ax=ax)
        ax.set_title(f"Ward dendrogram (cophenet = {c:.3f}, k={k})")
    return labels, Z, c


def dbscan_labels(X: np.ndarray, eps: float | None = None,
                  min_samples: int = 5) -> np.ndarray:
    """Lab IV DBSCAN.  If `eps` is None it is set to the 95th-percentile
    distance to the `min_samples`-th nearest neighbour — a common heuristic
    that approximates the k-distance elbow.
    """
    from sklearn.neighbors import NearestNeighbors

    if eps is None:
        nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
        d, _ = nn.kneighbors(X)
        eps = float(np.percentile(d[:, -1], 95))
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


def gmm_labels(X: np.ndarray, k: int, covariance_type: str = "full",
               random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Lab IV GMM.  Returns hard labels AND soft probabilities (n, k).

    Soft probabilities are what makes GMM more biologically realistic
    than KMeans for motor primitives — the worm transitions between
    primitives smoothly rather than snapping between hard labels.
    """
    gm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                         random_state=random_state).fit(X)
    return gm.predict(X), gm.predict_proba(X)


def compare_methods(X: np.ndarray, k: int = 4, random_state: int = 0) -> dict:
    """Run all four methods on the same X and return a comparison dict.

    Returns:
        labels: dict[method_name -> labels]
        silhouettes: dict[method_name -> silhouette_score]   (DBSCAN excluded if all noise)
        ari: pd.DataFrame                                    (pairwise ARI)
    """
    labels = {
        "KMeans":  kmeans_labels(X, k=k, random_state=random_state),
        "Ward":    ward_labels_and_dendrogram(X, k=k)[0],
        "DBSCAN":  dbscan_labels(X),
        "GMM":     gmm_labels(X, k=k, random_state=random_state)[0],
    }
    # silhouette requires >=2 labels and no all-noise
    sils = {}
    for name, lab in labels.items():
        uniq = set(int(x) for x in lab if x != -1)
        if len(uniq) >= 2:
            mask = lab != -1
            if mask.sum() > 1:
                sils[name] = float(silhouette_score(X[mask], lab[mask]))
    names = list(labels.keys())
    ari = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            ari.loc[a, b] = float(adjusted_rand_score(labels[a], labels[b]))
    return {"labels": labels, "silhouettes": sils, "ari": ari}


def choose_k_bic(X: np.ndarray, k_range=range(2, 8), random_state: int = 0):
    """Lab IV: choose k for GMM via BIC.  Returns (best_k, bic_values)."""
    bics = []
    for k in k_range:
        bics.append(GaussianMixture(n_components=k, covariance_type="full",
                                    random_state=random_state).fit(X).bic(X))
    bics = np.array(bics)
    return int(list(k_range)[int(bics.argmin())]), bics
