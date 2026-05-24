## ISSUE-020 — F1 reported at one tolerance only; add multi-tolerance + PR + ROC `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P2 |
| **Severity** | Completeness — single operating point hides the precision/recall trade-off |

**Description.** ISSUE-016 added `musical_f1(worm, chopin, tol_s=0.05)`. That is **one operating point** on what should be a curve. AppStat Lab VI standard practice is to show **F1 vs tolerance**, the **PR curve**, the **ROC curve**, and report AUC-PR / AUC-ROC. A step that happens to be optimal at exactly 50 ms tolerance may collapse at 25 ms; a step with a robust activation envelope will dominate across tolerances. None of this is visible from a single F1 number.

**Fix plan.** Add three functions to `pyannow/targets/midi_target.py`:

```python
def f1_vs_tolerance(worm_onsets, target_onsets, tols_s=(0.010, 0.025, 0.050, 0.100, 0.200, 0.400),
                    window_s=15.0) -> dict:
    """Return a dict tol_s -> musical_f1 result."""
    return {tol: musical_f1(worm_onsets, target_onsets, tol_s=tol, window_s=window_s)
            for tol in tols_s}

def precision_recall_curve_onsets(activation, target_onsets, t_arr, tol_s=0.05):
    """Treat the continuous activation envelope as a soft predictor at each 20ms bin;
    return (precision, recall, thresholds, AP).  Uses sklearn under the hood."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    y = _onsets_to_bin_labels(target_onsets, t_arr, tol_s=tol_s)
    p, r, thr = precision_recall_curve(y, activation)
    return p, r, thr, float(average_precision_score(y, activation))

def roc_curve_onsets(activation, target_onsets, t_arr, tol_s=0.05):
    """Same but the ROC curve.  Returns (fpr, tpr, thr, auc_roc)."""
    from sklearn.metrics import roc_auc_score, roc_curve
    y = _onsets_to_bin_labels(target_onsets, t_arr, tol_s=tol_s)
    fpr, tpr, thr = roc_curve(y, activation)
    return fpr, tpr, thr, float(roc_auc_score(y, activation))
```

Reference implementation already lives in `wormuse-analytics/src/wormuse_analytics/metrics.py` (`f1_vs_tolerance`) and `wormuse-analytics/src/wormuse_analytics/classification.py` (`precision_recall_curve_onsets`, `roc_curve_onsets`).

In notebook 03, add a new cell after cell 20:

```python
# F1 vs tolerance — across-step robustness
import pandas as pd
df_tol = pd.DataFrame()
for name, onsets in [('Step 0', onsets_base), ('Step 1', onsets_proc),
                      ('Step 2', onsets_clust), ('Step 3', onsets_ridge),
                      ('Step 4-6', onsets_mlp)]:
    d = f1_vs_tolerance(onsets, t_on_chopin, window_s=DURATION)
    for tol, r in d.items():
        df_tol = pd.concat([df_tol, pd.DataFrame([{'step': name, 'tol_ms': tol*1000, **r}])],
                           ignore_index=True)
# plot...
```

Plus a PR overlay using each step's activation envelope (Step 0 has none — leave blank).

**Affected files.**
- `PyANNOW/src/pyannow/targets/midi_target.py` — add three functions.
- `PyANNOW/tests/test_midi_target.py` — add tests (mirror existing `musical_f1` tests).
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — new cells after 20.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/TODO.md` — this entry.
