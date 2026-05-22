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


def procrustes_align(W_k: np.ndarray, C_k: np.ndarray) -> dict:
    """Compute the optimal orthogonal mapping from worm space to Chopin space.

    Parameters
    ----------
    W_k : (T, k_worm)   low-dim worm neural scores (rows = timesteps)
    C_k : (T, k_chopin) low-dim Chopin feature vectors

    Returns
    -------
    dict with keys:
        R          : (k_worm, k_chopin)  optimal rotation matrix
        residual   : float               ||W_k @ R - C_k||_F / ||C_k||_F
        singular_values : array          σ_i of C_k^T @ W_k (alignment quality)
    """
    # Step 1: compute the cross-covariance matrix  (L06: SVD setup)
    M = C_k.T @ W_k                          # (k_chopin, k_worm)

    # Step 2: SVD of M (L06: Eckart-Young)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    # Step 3: optimal rotation  R = V U^T  (L05: orthogonality, L06: proof)
    R = Vt.T @ U.T                            # (k_worm, k_chopin)

    # Step 4: evaluate alignment quality
    C_hat = W_k @ R
    residual = np.linalg.norm(C_k - C_hat, "fro") / (np.linalg.norm(C_k, "fro") + 1e-12)

    return {"R": R, "residual": float(residual), "singular_values": s}


def apply_rotation(W_k: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Map worm neural scores into the Chopin feature space."""
    return W_k @ R


def musical_distance(C_true: np.ndarray, C_pred: np.ndarray) -> float:
    """Normalised Frobenius distance between predicted and target features."""
    return float(np.linalg.norm(C_true - C_pred, "fro") /
                 (np.linalg.norm(C_true, "fro") + 1e-12))


def build_chopin_features(events: list, duration_s: float,
                           n_bins: int = 200, k_chopin: int = 8) -> np.ndarray:
    """Build a low-dim Chopin feature matrix from note events.

    Creates a T × k_chopin matrix where each row is the active-note indicator
    for that time bin, then compresses with PCA (= SVD of centred matrix,
    from L08) to k_chopin dimensions.

    Parameters
    ----------
    events     : list of NoteEvent (from midi_target.parse_midi)
    duration_s : clip length
    n_bins     : temporal resolution (rows of the output)
    k_chopin   : number of Chopin feature dimensions (columns)
    """
    from ..targets.midi_target import WORM_PITCHES  # noqa: F401
    dt = duration_s / n_bins
    pitches = sorted({int(e.pitch) for e in events if e.time_s <= duration_s})
    if not pitches:
        return np.zeros((n_bins, k_chopin))

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

    # Centre + truncated SVD to k_chopin (L08: PCA connection)
    roll -= roll.mean(axis=0)
    U, s, Vt = np.linalg.svd(roll, full_matrices=False)
    k = min(k_chopin, len(s))
    return (U[:, :k] * s[:k])      # (n_bins, k) scores
