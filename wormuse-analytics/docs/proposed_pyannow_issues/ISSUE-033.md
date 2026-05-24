## ISSUE-033 — Identical `find_peaks(height=mean)` peak detector across every step `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P1 |
| **Severity** | Correctness — every step uses the same magic threshold |

**Description.** Steps 1, 3, 4-6, 8 all share the same one-line onset detector:

```python
activ = np.abs(SOMETHING).max(axis=1)
peaks, _ = find_peaks(activ, distance=int(0.28/0.5e-3), height=activ.mean())
onsets = t_arr[peaks]
```

Different steps produce activations with different distributions — Ridge's `|C_pred|.max` is bounded, MLP's after-Adam-then-LBFGS is heavy-tailed, Procrustes' is on a standardised rotation scale. The single `height=activ.mean()` rule is a **magic threshold** that has no statistical justification and pulls each step to a different operating point.

Worse, the *number* of peaks chosen by `mean()` depends on the activation's skewness — heavily skewed activations (a few large peaks, many small) gate fewer peaks (`mean` is pulled up by the few large values), while symmetric activations gate more. So two steps with the same actual onset-detection quality can report very different F1 values simply because their activations are differently shaped.

**Fix plan.** Replace with a per-step calibrated classifier (see ISSUE-027). Schematically:

```python
def detect_onsets(activ, target_onsets, t_arr, refractory_s=0.28, tol_s=0.05):
    """Calibrated onset detector — closes ISSUE-033."""
    # 1. Build per-bin binary labels y from target_onsets within ±tol_s
    # 2. Fit LogisticRegression(class_weight='balanced')
    # 3. Pick threshold by Youden's J  (or best F1 on a held-out half)
    # 4. Enforce refractory period: greedy peak picking with min_distance.
    # ... reference impl: wormuse_analytics.classification.logistic_onset_detector
    return onsets
```

**Why a learned threshold matters specifically for this project.** PyANNOW's loss progression shows Step 0 winning. With the calibrated detector, *every* downstream step gets the threshold that maximises its own F1 — Step 0 (which doesn't go through `find_peaks`; it uses the body-wave's intrinsic note events) sees no change, while Steps 1-6 see their F1 rise. The ranking corrects itself.

**Caveat — leakage.** The calibrated detector tunes on Chopin onsets — same data the F1 is later computed on. The honest version splits Chopin into train/test halves: fit threshold on the first 5 s, evaluate F1 on the second 5 s. This connects to **ISSUE-038** (no cross-validation).

**Affected files.**
- `PyANNOW/src/pyannow/targets/midi_target.py` — add `detect_onsets` (reference: wormuse_analytics).
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — replace each `find_peaks` block with `detect_onsets`.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/TODO.md` — this entry. Depends on ISSUE-027 (logistic detector) and ISSUE-038 (CV).
