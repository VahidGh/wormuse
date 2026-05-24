## ISSUE-034 — Chopin features lossily compressed to k=8 — may discard >30% variance `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | ✅ Resolved — v0.8.0 |
| **Priority** | P2 |
| **Severity** | Correctness — logic problem #6 |

**Description.** `build_chopin_features(events, duration_s, k_chopin=8)` always truncated to exactly 8 PCA dimensions without reporting cumulative variance. For the Nocturne's 10 s window with 30+ unique pitches, k=8 captured only ~55-70% of variance. All supervised targets (Steps 1b, 3, 4-6, 8) were trained on a compressed C_chopin that lost meaningful pitch structure.

**Fix (v0.8.0).**

1. **Auto-k selection** (`k_chopin=None`, default): picks smallest k with cumvar ≥ `var_threshold` (default 90%).
2. **`chopin_cumvar(events, duration_s)`**: diagnostic function — returns the full cumulative variance curve for inspection in notebooks.
3. Explicit `k_chopin=N` still available for backward compatibility.

**Diagnostic (run in notebook 03 cell 4):**
```python
from pyannow.step1_svd.procrustes import chopin_cumvar
cv = chopin_cumvar(chopin_events, DURATION)
k8_cumvar = cv[7] if len(cv) > 7 else cv[-1]
k90       = int(np.searchsorted(cv, 0.90)) + 1
print(f"cumvar @ k=8: {k8_cumvar:.3f}")
print(f"k for 90% var: {k90}")
```

**Tested.** `test_step1_svd.py::TestBuildChopinFeatures::test_auto_k_selects_by_variance` verifies that auto-k satisfies cumvar ≥ 0.90.

**AppStat connection.** L01 PCA: the scree plot / cumulative variance curve is the standard tool for choosing the number of PCA components. k=8 was arbitrary; 90% rule is principled.

**Category:** `Category B — Data Pipeline`
