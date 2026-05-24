## ISSUE-034 — Chopin features lossily compressed to k=8 before training `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P2 |
| **Severity** | Correctness — supervised target carries less information than the worm could learn |

**Description.** `pyannow/step1_svd/procrustes.py:build_chopin_features`:

```python
def build_chopin_features(events, duration_s, n_bins=200, k_chopin=8):
    # binary piano roll  (T × n_pitches)
    ...
    roll -= roll.mean(axis=0)
    U, s, Vt = np.linalg.svd(roll, full_matrices=False)
    k = min(k_chopin, len(s))
    return (U[:, :k] * s[:k])   # (T, k) scores
```

The Chopin piano roll has `n_pitches ≈ 30-80` distinct pitches in the first 10 seconds. We compress to k=8 with PCA without ever reporting how much variance the truncation discards. Whatever Chopin information lives in dimensions ≥ 9 is invisible to every step that trains on `C_chopin` (Steps 1b, 3, 4-6, 8).

**Audit.** Run:

```python
sv = np.linalg.svd(roll, compute_uv=False)
cv = np.cumsum(sv**2) / (sv**2).sum()
print(f'cumvar @ k=8 : {cv[7]:.4f}')   # expected ≈ 0.5-0.7
print(f'k for 90% var: {int(np.searchsorted(cv, .9)+1)}')
```

If cumvar @ k=8 < 0.9, the supervised target itself caps every step's achievable F1 — independent of how well the worm encoder/composer works.

**Fix plan.**

1. **Compute and report the cumvar curve** of the piano roll in notebook 03 cell 4. If it's too low, raise k_chopin.

2. **Auto-select `k_chopin` by 90% rule** (default mode), with override:

   ```python
   def build_chopin_features(events, duration_s, n_bins=200, k_chopin=None,
                              var_threshold=0.9):
       """If k_chopin is None, pick the smallest k reaching var_threshold."""
       ...
   ```

3. **For RF and MLP** (which can handle high-dim outputs), fit on the **full piano roll** instead of the k-compressed scores. Ridge (which needs low dim for regularisation) keeps the compressed version. Add separate `C_chopin_full` and `C_chopin_k` variables.

**Affected files.**
- `PyANNOW/src/pyannow/step1_svd/procrustes.py` — modify `build_chopin_features`.
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — cumvar plot; `C_chopin_full` variable.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/tests/test_step1_svd.py` — test default k selection by var_threshold.
- `PyANNOW/TODO.md` — this entry.
