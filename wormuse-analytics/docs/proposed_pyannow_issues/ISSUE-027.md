## ISSUE-027 — Logistic regression baseline for onset detection `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P3 |
| **Severity** | Correctness — onset detector is a 1-feature classifier with a magic threshold |

**Description.** Every PyANNOW step from 1 onwards ends with:

```python
peaks, _ = find_peaks(activ, distance=int(0.28/0.5e-3), height=activ.mean())
```

This is **a 1-feature classifier with a hardcoded `mean(activ)` decision threshold** applied uniformly to every step's activation envelope. Different steps produce different envelope distributions; the same threshold cannot be optimal for all of them.

AppStat Lecture 05 covers exactly this case. The principled replacement is:

1. Fit `LogisticRegression(class_weight='balanced')` on `(activ_t, y_t)` where `y_t = 1` iff a Chopin onset is within ±50 ms of timestep `t`.
2. Pick the operating threshold by Youden's J (`TPR − FPR`) or best-F1, **not** the default 0.5.

This separates two questions that the current code conflates:
- *Did the activation contain the signal?* (AUC tells us)
- *Did we pick the right threshold?* (Youden tells us)

**Fix plan.** Add to `pyannow/targets/midi_target.py`:

```python
def logistic_onset_detector(activation, target_onsets, t_arr,
                            tol_s=0.05, threshold_strategy='youden'):
    """Calibrated logistic onset detector + threshold tuning."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
    # ... see wormuse-analytics/src/wormuse_analytics/classification.logistic_onset_detector
```

In notebook 03, after each step that produces an activation envelope, add:

```python
res = logistic_onset_detector(activ_step1, t_on_chopin, t_arr, threshold_strategy='youden')
print(f'  calibrated F1 = {res["f1"]:.3f}  threshold = {res["threshold"]:.4f}')
```

The **gap** between `res["f1"]` and the find_peaks-based F1 quantifies how much of each step's poor score was caused by the magic threshold versus the activation itself. Many steps will rebound substantially under calibrated thresholding.

**Affected files.**
- `PyANNOW/src/pyannow/targets/midi_target.py` — add `logistic_onset_detector`.
- `PyANNOW/tests/test_midi_target.py` — add reproducibility test.
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — calibrated F1 reported per step.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/TODO.md` — this entry.
