"""Step 2 — Motor primitive discovery via PCA + K-means.

NAML material:  Lecture 08 (PCA — projection onto maximum-variance directions)
                Lecture 10 (K-means clustering)
                Lab 02     (PCA on MNIST; cancer diagnostic)
                Lab 03     (K-means-like analysis; PageRank as bonus idea)

The locomotion circuit of C. elegans switches between a small number of
discrete "motor primitives" — forward crawl, backward crawl, Ω-turn.
We discover these automatically from the worm's neural activity:

  1. PCA of the neural trajectory → top-2 PCs capture >80% of variance
     (L08: the k PCs are the k directions of maximum variance in ℝ^302)
  2. K-means on the PC scores → 4 clusters ≈ motor primitives
     (L10: K-means minimises within-cluster sum of squares)
  3. Each cluster centroid is assigned a musical note category

This is identical in structure to the Lab02 cancer diagnostic:
    Lab02: PCA(MNIST) → KNN classifier → accuracy
    Here:  PCA(neural) → K-means → musical category assignment
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Map cluster → pentatonic pitch (D♭ major, matching Chopin's key)
CLUSTER_TO_PITCH = {0: 61, 1: 63, 2: 65, 3: 68,   # 4 main clusters
                    4: 70, 5: 73, 6: 75, 7: 78}    # optional extra


def pca_reduce(X: np.ndarray, n_components: int = 4,
               standardize: bool = True) -> tuple[np.ndarray, PCA]:
    """PCA of neural activity matrix X ∈ ℝ^{T × 302}.

    Mirrors Lab02 pattern: StandardScaler + PCA.
    Returns (scores T×k, fitted PCA object).

    NAML L08 connection:
        PCA maximises variance along successive orthogonal directions.
        The variance ratio tells us how much of the worm's movement is
        explained by k dimensions — analogous to the cancer Lab's scree plot.
    """
    Xt = X.T if X.shape[0] > X.shape[1] else X   # ensure (T, 302)
    if standardize:
        mu  = Xt.mean(axis=0)
        std = Xt.std(axis=0) + 1e-10
        Xt  = (Xt - mu) / std
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(Xt)
    return scores, pca


def find_motor_primitives(scores: np.ndarray, k: int = 4,
                           random_state: int = 0) -> tuple[np.ndarray, KMeans]:
    """K-means on PCA scores to discover motor primitive clusters.

    Lab02 analogy: we cluster neural states just as we clustered digit images.

    Returns (labels T-vector, fitted KMeans).
    """
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(scores)
    return labels, km


def choose_k_silhouette(scores: np.ndarray, k_range: range = range(2, 8)) -> dict:
    """Choose the best number of clusters by silhouette score.

    NAML L10 connection:  the silhouette score measures cluster cohesion
    (analogous to within-cluster SS) vs separation.
    """
    sils = {}
    for k in k_range:
        labels, _ = find_motor_primitives(scores, k=k)
        if len(np.unique(labels)) < 2:
            continue
        sils[k] = silhouette_score(scores, labels)
    best_k = max(sils, key=sils.get) if sils else 4
    return {"scores": sils, "best_k": best_k}


def cluster_to_notes(labels: np.ndarray,
                      t_arr_ms: np.ndarray,
                      min_interval_ms: float = 280.0) -> list:
    """Convert per-timestep cluster labels to note events.

    Each transition into a new cluster = one note onset.
    Enforces the BWM refractory constraint (min_interval_ms).

    Returns: list of (time_s, pitch_midi, velocity)
    """
    events = []
    prev_label = -1
    last_t = -min_interval_ms

    for i, (t, lab) in enumerate(zip(t_arr_ms, labels)):
        if lab != prev_label and (t - last_t) >= min_interval_ms:
            pitch = CLUSTER_TO_PITCH.get(int(lab), 61)
            # Velocity proportional to cluster distinctness (simple proxy)
            vel = min(64 + int(lab) * 10, 127)
            events.append((float(t) * 1e-3, pitch, vel))
            last_t = t
        prev_label = lab

    return events
