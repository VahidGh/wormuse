## ISSUE-018 — Builder/notebook desync + synthetic neural rank collapse `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | ✅ Resolved — commits `1dbd3f7`, `12cfd80` (v0.7.0) |
| **Priority** | P0 |
| **Severity** | Architecture — root cause of all NAML steps scoring worse than Step 0 |

**Description.** Two compounding problems prevented any NAML learning step from beating Step 0:

1. **Synthetic X_neural rank=1**: `X_neural` was `np.vstack([V_mus.T]*302)[:302]` — 302 copies of the 96 muscle signals → rank 1. SVD found k=1 PC; Ridge/MLP/L-BFGS could not learn any meaningful mapping.

2. **Builder/notebook desync**: `_build_naml_progression_nb.py` and the executed notebook had accumulated drift, making comparisons unreliable.

**Resolution (v0.7.0):**
- `generate_neural_activity_302()` added to `worm_optimizer.py` — biologically-structured 302-neuron matrix with k≥4 independent PCs (cmd interneurons, A/B/D motor neurons, sensory, body groups per Boyle et al. 2012).
- Notebook 03 rebuilt and re-executed with `generate_neural_activity_302()`.
- 96-cell Boyle 4×24 model is now the default.
- 40 new tests in `test_forward_model.py`.

**Impact on AppStat issues:** Closes ISSUE-029 (logic problem #1 — synthetic neural rank). The k≥4 PCs now enable ISSUE-023 (PCA biplot) to show meaningful structure.

**Category:** `Category D — Architecture & Data Pipeline`
