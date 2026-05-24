## ISSUE-020 — Single-tolerance F1 only; no PR curve, no multi-tol analysis `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | ✅ Resolved — v0.8.0 |
| **Priority** | P2 |
| **Severity** | Incomplete evaluation — AppStat L06 minimum |

**Description.** `musical_f1` reported a single point estimate at 50 ms tolerance. A model that achieves F1=0.20 at 50 ms may have F1=0.35 at 100 ms — crucial information for understanding temporal precision vs. rhythmic quality trade-offs.

**Fix (v0.8.0).** `precision_recall_at_tolerances()` in `midi_target.py`:

```python
results = precision_recall_at_tolerances(
    worm_onsets, chopin_onsets,
    tols=(0.025, 0.05, 0.10, 0.20),
    window_s=15.0,
)
# Returns: [{"tol_s": 0.025, "precision": ..., "recall": ..., "f1": ...}, ...]
```

Usage in notebook 03 cell 20: plot F1 vs. tolerance for each step on a single axes — a multi-tolerance F1 curve analogous to a precision-recall curve but parameterised by temporal tolerance.

**Tested.** `test_midi_target.py::TestPrecisionRecallAtTolerances`:
- Correct number of entries per tolerance list
- F1 monotone non-decreasing with wider tolerance
- Perfect match → F1=1.0 at all tolerances

**AppStat connection.** L06 ROC/PR curves: sweeping a threshold parameter reveals the full operating characteristic. Here the "threshold" is temporal tolerance.

**Category:** `Category A — Metrics & Scoring`
