# PyANNOW AppStat Issues — Grouped by Test Category

> Generated for v0.8.0 — updated to reflect 96-cell Boyle architecture (v0.7.0).
> Each category maps to one pytest test file so the full audit can be run as a single `pytest` invocation.

---

## Run all categories at once

```bash
PYTHONPATH=PyANNOW/src python3 -m pytest \
    PyANNOW/tests/test_midi_target.py \
    PyANNOW/tests/test_step1_svd.py \
    PyANNOW/tests/test_training_cv.py \
    PyANNOW/tests/test_forward_model.py \
    -v --tb=short -q
```

---

## Category A — Metrics & Scoring
**Test file:** `PyANNOW/tests/test_midi_target.py`
**AppStat lectures:** L05 (F1, classification), L06 (ROC/PR, bootstrap)

| Issue | Title | Status | v |
|---|---|---|---|
| ISSUE-016 | onset_loss gameable → musical_f1 + ioi_similarity | ✅ | 0.6 |
| ISSUE-017 | per-step F1 tracking | ✅ | 0.6 |
| ISSUE-019 | Cell 20 left panel misleads (onset_loss prominent) | 🔴 | — |
| ISSUE-020 | Multi-tol F1 curve (precision_recall_at_tolerances) | ✅ | **0.8** |
| ISSUE-021 | Bootstrap CIs for F1 (bootstrap_musical_f1) | ✅ | **0.8** |
| ISSUE-035 | Pitch-aware F1 (time + pitch matching) | ✅ | **0.8** |
| ISSUE-037 | Biological pitch ceiling (biological_pitch_ceiling) | ✅ | **0.8** |

**New test classes (v0.8.0):** `TestPitchAwareF1`, `TestBiologicalPitchCeiling`, `TestPrecisionRecallAtTolerances`, `TestBootstrapMusicalF1`

---

## Category B — Data Pipeline
**Test file:** `PyANNOW/tests/test_step1_svd.py`
**AppStat lectures:** L01 (PCA/SVD), L04 (standardization)

| Issue | Title | Status | v |
|---|---|---|---|
| ISSUE-032 | Procrustes unstandardized (z-score W_k) | ✅ | **0.8** |
| ISSUE-034 | Lossy k=8 Chopin compression (auto-k 90% rule) | ✅ | **0.8** |

**New test classes (v0.8.0):** `TestBuildChopinFeatures`, `TestChopinCumvar`, `TestProcrustesAlign`

---

## Category C — Statistical Validation
**Test file:** `PyANNOW/tests/test_training_cv.py`
**AppStat lectures:** L06 (cross-validation, bootstrap), L04 (Durbin-Watson)

| Issue | Title | Status | v |
|---|---|---|---|
| ISSUE-026 | Durbin-Watson on Ridge residuals | 🔴 | — |
| ISSUE-027 | Logistic onset detector (replace magic threshold) | 🔴 | — |
| ISSUE-028 | RandomForest baseline | 🔴 | — |
| ISSUE-038 | Time-series CV (time_series_cv + blocked_bootstrap_ci) | ✅ | **0.8** |

**New test classes (v0.8.0):** `TestTimeSeriesCV`, `TestBlockedBootstrapCI`

---

## Category D — Architecture & Data Pipeline
**Test file:** `PyANNOW/tests/test_forward_model.py`
**AppStat lectures:** L01 (PCA rank), L00 (data quality)

| Issue | Title | Status | v |
|---|---|---|---|
| ISSUE-018 | Builder/notebook desync + rank-1 neural data | ✅ | 0.7 |
| ISSUE-029 | Synthetic 302-neuron rank≥4 (generate_neural_activity_302) | ✅ | 0.7 |
| ISSUE-030 | Step 0 mislabeled as "ML" step | 🔴 | — |
| ISSUE-031 | 8-muscle pitch bottleneck (96-cell Boyle model) | ✅ | 0.7 |
| ISSUE-033 | Magic-threshold onset detector | 🔴 | — |
| ISSUE-036 | Step 8 PINN identity (doc note added) | ✅ | **0.8** |

---

## Category E — Visualization & Notebook Cells
**Test file:** *(notebook-level — no automated unit tests)*
**AppStat lectures:** L00-L03

| Issue | Title | Status | Effort |
|---|---|---|---|
| ISSUE-019 | Cell 20 left panel redesign | 🔴 | medium |
| ISSUE-022 | Descriptive stats + IOI KDE | 🔴 | medium |
| ISSUE-023 | PCA biplot of worm neural subspace | 🔴 | low |
| ISSUE-024 | t-SNE/UMAP motor-state manifold | 🔴 | low |
| ISSUE-025 | Four clustering methods comparison | 🔴 | medium |

---

## Status summary

| Category | Issues | ✅ Resolved | 🔴 Open |
|---|---|---|---|
| A — Metrics & Scoring | 7 | 6 | 1 (ISSUE-019) |
| B — Data Pipeline | 2 | 2 | 0 |
| C — Statistical Validation | 4 | 1 | 3 |
| D — Architecture | 6 | 5 | 1 (ISSUE-033) |
| E — Visualization | 5 | 0 | 5 |
| **Total** | **24** | **14** | **10** |

---

## v0.8.0 new code summary

| File | New functions | Fixes |
|---|---|---|
| `pyannow/targets/midi_target.py` | `pitch_aware_f1`, `biological_pitch_ceiling`, `precision_recall_at_tolerances`, `bootstrap_musical_f1` | ISSUE-035, 037, 020, 021 |
| `pyannow/step1_svd/procrustes.py` | `chopin_cumvar`; modified `build_chopin_features` + `procrustes_align` | ISSUE-034, 032 |
| `pyannow/training/cv.py` | `time_series_cv`, `blocked_bootstrap_ci` | ISSUE-038 |
| `pyannow/step8_pinn/locomotion_pinn.py` | arch note box | ISSUE-036 |
| `PyANNOW/tests/test_midi_target.py` | 15 new test methods | ISSUE-020, 021, 035, 037 |
| `PyANNOW/tests/test_step1_svd.py` | 12 new test methods | ISSUE-032, 034 |
| `PyANNOW/tests/test_training_cv.py` | 12 new test methods | ISSUE-038 |
