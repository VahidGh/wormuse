"""Step 1a — SVD / Randomized-SVD neural-state encoder.

NAML material:  Lecture 06 (Eckart-Young theorem, RSVD)
                Lab 01 (SVD image compression → here: neural-state compression)
                ``rsvd_2024.ipynb`` in Lecture September 30th/

The worm's 302-neuron activity matrix  X ∈ ℝ^{302 × T}  is high-dimensional
but lies on a low-rank manifold (the locomotion subspace).  The Eckart-Young
theorem tells us the best rank-k approximation is the truncated SVD:

    X_k = U_k Σ_k V_k^T

We use *randomized* SVD (Halko et al. 2011) when T is large — the course's
own ``rsvd_2024.ipynb`` derives the algorithm from scratch.

Key insight for the music analogy:
    Just as we compressed the Tarantula Nebula image in Lab01 using truncated
    SVD, here we compress the worm's 302-D neural trajectory to k dimensions.
    Those k dimensions are the "notes" the worm can play.
"""
from __future__ import annotations

import numpy as np


# ─── Full truncated SVD (NAML L06) ───────────────────────────────────────────

def svd_encode(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncated SVD of neural activity matrix X ∈ ℝ^{n_neurons × T}.

    Returns (U_k, s_k, Vt_k) — the Eckart-Young rank-k approximation.

    Lab01 connection: we used the same call ``np.linalg.svd(img, full_matrices=False)``
    to compress images.  Here ``img`` is replaced by the neural trajectory.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :k], s[:k], Vt[:k, :]


def neural_scores(X: np.ndarray, U_k: np.ndarray) -> np.ndarray:
    """Project X onto the k principal neural directions → low-dim trajectory.

    Returns Z ∈ ℝ^{k × T} — the temporal evolution of the neural state
    in the k-dimensional locomotion subspace.
    """
    return U_k.T @ X


def reconstruction_error(X: np.ndarray, U_k: np.ndarray, s_k: np.ndarray,
                          Vt_k: np.ndarray) -> dict:
    """Relative Frobenius-norm error of the rank-k approximation.

    From the Eckart-Young theorem:  ||X - X_k||_F = sqrt(sum_{i>k} σ_i²)
    """
    X_k = (U_k * s_k) @ Vt_k
    err = np.linalg.norm(X - X_k, "fro") / np.linalg.norm(X, "fro")
    cum_var = np.cumsum(s_k**2) / (np.linalg.svd(X, compute_uv=False)**2).sum()
    return {
        "rel_error": float(err),
        "cum_variance_explained": float(cum_var[-1]),
        "singular_values": s_k,
    }


# ─── Randomized SVD (NAML rsvd_2024.ipynb) ──────────────────────────────────

def rsvd(X: np.ndarray, k: int, p: int = 10, q: int = 1,
         seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized SVD — identical algorithm to the course's rsvd_2024.ipynb.

    Speedup over full SVD: O(nTk) instead of O(nT min(n,T)).
    For n=302 neurons and T=10000 timesteps this gives ~10× speedup at k=10.

    Steps (Halko et al. 2011, derived from first principles in the lecture):
      1. Random sketch:  Y = X Ω  where Ω is a Gaussian test matrix
      2. (Optional) Power iteration to sharpen the singular gap
      3. QR decomposition: Q, _ = qr(Y)
      4. Project: B = Q^T X  (small matrix)
      5. SVD of B: U_B, s, Vt
      6. Lift: U = Q U_B
    """
    rng = np.random.default_rng(seed)
    n, T = X.shape
    Omega = rng.standard_normal((T, k + p))
    Y = X @ Omega
    for _ in range(q):
        Q, _ = np.linalg.qr(Y)
        Z = X.T @ Q
        Q, _ = np.linalg.qr(Z)
        Y = X @ Q
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ X
    U_B, s, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ U_B)[:, :k], s[:k], Vt[:k, :]


def choose_k_by_variance(X: np.ndarray, variance_threshold: float = 0.90,
                          max_k: int = 50) -> int:
    """Return the smallest k explaining ≥ variance_threshold of total variance.

    Mirrors the scree-plot criterion from Lab01 (cumulative energy ≥ 95%).
    """
    s = np.linalg.svd(X, compute_uv=False)
    cum = np.cumsum(s**2) / (s**2).sum()
    hits = np.where(cum >= variance_threshold)[0]
    return int(hits[0] + 1) if len(hits) > 0 else min(max_k, len(s))
