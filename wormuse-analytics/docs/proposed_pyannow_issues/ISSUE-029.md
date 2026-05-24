## ISSUE-029 — Synthetic 302-neuron matrix rank=1 prevents all NAML learning `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | ✅ Resolved — commit `1dbd3f7` (v0.7.0, part of ISSUE-018) |
| **Priority** | P1 |
| **Severity** | Architecture — logic problem #1 |

**Description.** `X_neural` was generated as `np.vstack([V_mus.T] * n)[:302]` — 302 repetitions of muscle signals, collapsing to rank 1. SVD step 1 found k=1 PC. No NAML learning step could outperform Step 0.

**Resolution (v0.7.0).** `generate_neural_activity_302()` in `worm_optimizer.py`:
- 302 neurons structured in 7 biologically-motivated groups (Boyle et al. 2012, White et al. 1986)
- k≥4 independent PCs guaranteed
- Deterministic (seed parameter); shape/rank tests in `test_forward_model.py`

**AppStat connection.** L01 PCA biplot (ISSUE-023) now shows 4+ meaningful directions; Ridge (step 3) can learn a non-trivial mapping.

**Category:** `Category D — Architecture & Data Pipeline`
