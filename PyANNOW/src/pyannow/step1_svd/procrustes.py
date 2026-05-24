"""Step 1b — Orthogonal Procrustes alignment (SVD-based transfer learning).

NAML material:  Lecture 06 (SVD, Eckart-Young)
                Lecture 09 (Pseudoinverse, least squares)
                Lecture 05 (Projection matrix, orthogonality)

The Procrustes problem:
    Given low-dim worm scores  W ∈ ℝ^{T × k_w}
    and  low-dim Chopin features  C ∈ ℝ^{T × k_c},
    find the orthogonal matrix  R ∈ ℝ^{k_w × k_c}  that minimises

        ||W R - C||_F²

    subject to  R^T R = I (or the closest possible).

Solution via SVD of  C^T W:
    M = C^T W  →  M = U Σ V^T  →  R = V U^T

This is the **closest orthogonal matrix** to the unconstrained least-squares
solution.  It is guaranteed to be the global minimum by the Eckart-Young theorem
applied to the bi-orthogonal factor problem.

Music analogy:
    We want to *rotate* the worm's neural subspace to face the same direction as
    Chopin's musical subspace.  If they were identical, R would be the identity.
    The residual  ||W R - C||_F  measures how different the two "languages" are.
"""
from __future__ import annotations

import numpy as np


def procrustes_align(W_k: np.ndarray, C_k: np.ndarray,
                     standardize: bool = True) -> dict:
    """Compute the optimal orthogonal mapping from worm space to Chopin space.

    Parameters
    ----------
    W_k : (T, k_worm)   low-dim worm neural scores (rows = timesteps)
    C_k : (T, k_chopin) low-dim Chopin feature vectors
    standardize : if True (default), z-score W_k column-wise before alignment.
        This is required when the worm PCA scores and Chopin scores have
        very different scales (ISSUE-032): without standardization the first
        PC dominates the Frobenius norm and the rotation R is poorly
        conditioned.  The scale factors are stored in `scale` for inversion.

    Returns
    -------
    dict with keys:
        R          : (k_worm, k_chopin)  optimal rotation matrix
        residual   : float               ||W_k_scaled @ R - C_k||_F / ||C_k||_F
        singular_values : array          σ_i of C_k^T @ W_k (alignment quality)
        scale      : (k_worm,) std of each worm column (1.0 if standardize=False)
        W_k_scaled : (T, k_worm) standardized worm scores (input to SVD)
    """
    W = np.array(W_k, dtype=float)

    if standardize:
        scale = W.std(axis=0)
        scale[scale < 1e-12] = 1.0        # avoid division by zero for dead PCs
        W = W / scale
    else:
        scale = np.ones(W.shape[1])

    # Step 1: compute the cross-covariance matrix  (L06: SVD setup)
    M = C_k.T @ W                           # (k_chopin, k_worm)

    # Step 2: SVD of M (L06: Eckart-Young)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    # Step 3: optimal rotation  R = V U^T  (L05: orthogonality, L06: proof)
    R = Vt.T @ U.T                          # (k_worm, k_chopin)

    # Step 4: evaluate alignment quality
    C_hat = W @ R
    residual = np.linalg.norm(C_k - C_hat, "fro") / (np.linalg.norm(C_k, "fro") + 1e-12)

    return {"R": R, "residual": float(residual), "singular_values": s,
            "scale": scale, "W_k_scaled": W}


def apply_rotation(W_k: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Map worm neural scores into the Chopin feature space."""
    return W_k @ R


def musical_distance(C_true: np.ndarray, C_pred: np.ndarray) -> float:
    """Normalised Frobenius distance between predicted and target features."""
    return float(np.linalg.norm(C_true - C_pred, "fro") /
                 (np.linalg.norm(C_true, "fro") + 1e-12))


def build_chopin_features(
    events:        list,
    duration_s:    float,
    n_bins:        int   = 200,
    k_chopin:      int | None = None,
    var_threshold: float = 0.90,
) -> np.ndarray:
    """Build a low-dim Chopin feature matrix from note events.

    Creates a T × k matrix where each row is the active-note indicator for
    that time bin, then compresses with PCA (= SVD of centred matrix, L08)
    to k_chopin (or auto-selected k) dimensions.

    Parameters
    ----------
    events        : list of NoteEvent (from midi_target.parse_midi)
    duration_s    : clip length in seconds
    n_bins        : temporal resolution (rows of output)
    k_chopin      : explicit number of Chopin PCA dimensions.  If None
                    (default), automatically selects the smallest k that
                    explains ≥ var_threshold of variance (ISSUE-034 fix).
    var_threshold : target cumulative variance ratio (used only when
                    k_chopin is None).  Default 0.90 = 90%.

    Returns
    -------
    C : (n_bins, k)  Chopin PCA scores matrix
    """
    dt = duration_s / n_bins
    pitches = sorted({int(e.pitch) for e in events if e.time_s <= duration_s})
    if not pitches:
        k_fallback = k_chopin if k_chopin is not None else 8
        return np.zeros((n_bins, k_fallback))

    # Binary piano roll  (n_bins × n_pitches)
    roll = np.zeros((n_bins, len(pitches)), dtype=float)
    pitch_idx = {p: i for i, p in enumerate(pitches)}
    for ev in events:
        if ev.time_s > duration_s:
            break
        t0 = int(ev.time_s / dt)
        t1 = min(int((ev.time_s + max(ev.duration, dt)) / dt) + 1, n_bins)
        pi = pitch_idx.get(int(ev.pitch))
        if pi is not None:
            roll[t0:t1, pi] = float(ev.velocity) / 127.0

    # Centre + full SVD for variance diagnostics  (L08: PCA connection)
    roll -= roll.mean(axis=0)
    U, s, Vt = np.linalg.svd(roll, full_matrices=False)

    # Determine k
    if k_chopin is None:
        sv2 = s ** 2
        cumvar = np.cumsum(sv2) / (sv2.sum() + 1e-12)
        k = int(np.searchsorted(cumvar, var_threshold)) + 1
        k = max(1, min(k, len(s)))
    else:
        k = min(k_chopin, len(s))

    return (U[:, :k] * s[:k])      # (n_bins, k) scores


def chopin_cumvar(
    events:     list,
    duration_s: float,
    n_bins:     int = 200,
) -> np.ndarray:
    """Return the cumulative variance explained curve of the Chopin piano roll.

    Useful for diagnosing how many PCA dimensions are needed before truncating
    (ISSUE-034 diagnostic).  Returns a (n_pitches,) array of cumulative variance
    fractions.

    Example usage::

        cv = chopin_cumvar(events, 10.0)
        print(f"cumvar @ k=8 : {cv[7]:.4f}")
        print(f"k for 90% var: {int(np.searchsorted(cv, .9)) + 1}")
    """
    dt = duration_s / n_bins
    pitches = sorted({int(e.pitch) for e in events if e.time_s <= duration_s})
    if not pitches:
        return np.array([1.0])

    roll = np.zeros((n_bins, len(pitches)), dtype=float)
    pitch_idx = {p: i for i, p in enumerate(pitches)}
    for ev in events:
        if ev.time_s > duration_s:
            break
        t0 = int(ev.time_s / dt)
        t1 = min(int((ev.time_s + max(ev.duration, dt)) / dt) + 1, n_bins)
        pi = pitch_idx.get(int(ev.pitch))
        if pi is not None:
            roll[t0:t1, pi] = float(ev.velocity) / 127.0

    roll -= roll.mean(axis=0)
    sv = np.linalg.svd(roll, compute_uv=False)
    sv2 = sv ** 2
    return np.cumsum(sv2) / (sv2.sum() + 1e-12)
