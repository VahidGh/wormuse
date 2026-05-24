## ISSUE-032 — Procrustes alignment between unstandardized incommensurate spaces `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P2 |
| **Severity** | Correctness — the rotation has no physical meaning as currently set up |

**Description.** Step 1b solves the orthogonal Procrustes problem:

```
R* = argmin_{R^T R = I} || Z_worm @ R - C_chopin ||_F
```

with:
- `Z_worm ∈ ℝ^{T × 4}` — PCA scores of the synthetic 302-D noisy-muscle matrix
- `C_chopin ∈ ℝ^{T × 8}` — PCA scores of a binary Chopin piano roll

These two spaces have:
- Different units (muscle voltage vs note-on indicator).
- Different scales (the PCA scores inherit variance from their parent matrices).
- Different feature semantics (motor activity vs musical structure).
- No reason to believe they share an orthonormal basis at all.

Without standardization, the "rotation" R found by SVD of `C^T @ W` minimises a Frobenius distance dominated by whichever space happens to have larger numerical variance — a numerical accident, not a physical alignment.

AppStat Lab II / V principle: **standardize features before any linear comparison or fitting.**

**Fix plan.**

1. In `pyannow/step1_svd/procrustes.py:procrustes_align`, add standardization:

   ```python
   def procrustes_align(W_k, C_k, standardize=True):
       if standardize:
           W_k = (W_k - W_k.mean(axis=0)) / (W_k.std(axis=0) + 1e-12)
           C_k = (C_k - C_k.mean(axis=0)) / (C_k.std(axis=0) + 1e-12)
       M = C_k.T @ W_k
       U, s, Vt = np.linalg.svd(M, full_matrices=False)
       R = Vt.T @ U.T
       residual = np.linalg.norm(C_k - W_k @ R, "fro") / (np.linalg.norm(C_k, "fro") + 1e-12)
       return {"R": R, "residual": float(residual), "singular_values": s}
   ```

2. The `R` returned now relates standardized worm and Chopin scores — interpretable as a relative orientation between the two manifolds rather than an unscaled rotation.

3. Document the change in cell 9 markdown: "We standardize both spaces before Procrustes because they have different units (muscle voltage vs piano-roll indicator); the rotation R then expresses the orientation between standardized manifolds."

**Affected files.**
- `PyANNOW/src/pyannow/step1_svd/procrustes.py` — add `standardize=True` param, default True.
- `PyANNOW/tests/test_step1_svd.py` — test that pre-standardized inputs give R ≈ identity.
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — cell 9 markdown.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/TODO.md` — this entry.
