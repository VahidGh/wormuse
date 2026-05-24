## ISSUE-023 — Add PCA biplot + consistent standardization `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P3 |
| **Severity** | Interpretability — PCA produces opaque scores; no biplot |

**Description.** Step 1a (RSVD) and Step 2 (PCA) compress the 302-neuron activity into 4 components but never show **which neurons drive PC1 vs PC2** — the biplot is the AppStat Lab II tool that makes PCA interpretable. Additionally, Step 1a applies RSVD to the raw `X_neural` matrix while Step 2 calls `pca_reduce(..., standardize=True)`. The two steps see different feature spaces, which is inconsistent and worth documenting (or, ideally, fixing).

**Fix plan.**

1. **Biplot helper.** Add to `pyannow/step1_svd/encoder.py` (or `pyannow/utils/plots.py` if such a module exists):

   ```python
   def biplot(Z, components, ax=None, feature_names=None, sample_colors=None,
              arrow_scale=5.0, max_arrows=15):
       """Lab II biplot — scores scatter + top loading arrows."""
       # see wormuse-analytics/src/wormuse_analytics/dimreduction.py for the reference impl
   ```

2. **Notebook cell after Step 1a:** colour each time-bin's PC1/PC2 score by its dominant muscle index, overlay the top-magnitude loading arrows. This visualises the 8-muscle block structure of the synthetic 302-neuron matrix.

3. **Standardization consistency.** Either (a) standardise the input to RSVD in Step 1a too, or (b) document explicitly in the markdown why Step 1a's "compress raw activity" and Step 2's "standardise then compress" are intentionally different (Step 1a's downstream Procrustes alignment is scale-aware; Step 2's KMeans is not).

**Affected files.**
- `PyANNOW/src/pyannow/step1_svd/encoder.py` (or `utils/plots.py`) — `biplot` helper.
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — new cell after Step 1a.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/TODO.md` — this entry.
