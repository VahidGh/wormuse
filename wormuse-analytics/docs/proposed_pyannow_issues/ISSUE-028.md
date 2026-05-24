## ISSUE-028 — RandomForest baseline + permutation importance `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P2 |
| **Severity** | Validation — no model-agnostic ceiling for Steps 4-6 to be compared against |

**Description.** The NAML pipeline escalates Linear (Ridge) → MLP (Adam) → L-BFGS → PINN, a deep-learning ramp. AppStat Lecture 07 says: **fit a Random Forest first, look at the F1, decide whether you need the deep model.** If RF F1 matches MLP F1, the deep model is over-engineering. If RF F1 falls clearly short, the MLP earns its place.

Currently no such ceiling is computed; the MLP wins or loses by inspection, with no model-agnostic comparison.

**Fix plan.** Add a new step (Step 7) — `pyannow/step7_trees/rf_baseline.py`:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

def rf_baseline(Z, C, n_estimators=500, random_state=0):
    Zs = StandardScaler().fit_transform(Z)
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1,
                                oob_score=True, random_state=random_state).fit(Zs, C)
    pi = permutation_importance(rf, Zs, C, n_repeats=10, n_jobs=-1,
                                 random_state=random_state)
    return rf, pi
```

In notebook 03, add a cell between Step 3 (Ridge) and Steps 4-6 (MLP):

```python
rf, pi = rf_baseline(Z_worm, C_chopin)
C_pred_rf = rf.predict(StandardScaler().fit_transform(Z_worm))
activ_rf = np.abs(C_pred_rf).max(axis=1)
peaks_rf, _ = find_peaks(activ_rf, distance=int(0.28/0.5e-3), height=activ_rf.mean())
onsets_rf = t_arr[peaks_rf]
L_rf = onset_loss(onsets_rf, t_on_chopin, window_s=DURATION)
f1_rf = musical_f1(onsets_rf, t_on_chopin, window_s=DURATION)
print(f'RF baseline: F1={f1_rf["f1"]:.3f}  R^2_OOB={rf.oob_score_:.3f}')
losses['Step 7 (RF baseline)'] = L_rf
f1_scores['Step 7 (RF baseline)'] = f1_rf['f1']
```

Then show **permutation feature importance** per PC — confirms or refutes the k=4 hardcoded dimensionality in a model-agnostic way.

**Affected files.**
- `PyANNOW/src/pyannow/step7_trees/rf_baseline.py` — new module.
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — new cell + permutation-importance plot.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/pyproject.toml` — sklearn already a dep; no change.
- `PyANNOW/TODO.md` — this entry.
