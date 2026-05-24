## ISSUE-032 — Procrustes operates on un-standardized columns — first PC dominates rotation `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | ✅ Resolved — v0.8.0 |
| **Priority** | P2 |
| **Severity** | Numerical — logic problem #4 |

**Description.** `procrustes_align(W_k, C_k)` minimises `||W_k R - C_k||_F²` without standardizing `W_k`. With the 96-cell Boyle model, PC1 has amplitude ~10× larger than PC4. The Frobenius norm is dominated by PC1, and R is essentially a rank-1 rotation. PCs 2-4 are poorly aligned regardless of learning quality.

**Fix (v0.8.0).** `procrustes_align(W_k, C_k, standardize=True)` (new default):
- Z-score each column of `W_k` before computing `M = C_k^T @ W_k`
- Scale factors stored in `result["scale"]` for downstream inversion
- Standardized `W_k_scaled` stored in `result["W_k_scaled"]`
- `standardize=False` available for backward compatibility

**Tested.** `test_step1_svd.py::TestProcrustesAlign::test_standardize_reduces_residual_on_unscaled_input` confirms that standardized residual ≤ raw residual for ill-scaled W.

**AppStat connection.** AppStat L04 OLS diagnostics: standardizing predictors is a prerequisite for interpreting regression coefficients and comparing their magnitudes. Same principle applies here.

**Category:** `Category B — Data Pipeline`
