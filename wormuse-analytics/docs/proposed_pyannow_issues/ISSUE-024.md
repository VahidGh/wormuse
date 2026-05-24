## ISSUE-024 — Add t-SNE / UMAP of motor-state manifold `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P3 |
| **Severity** | Validation — KMeans labels never validated in a nonlinear projection |

**Description.** PCA is linear; the worm's actual neural manifold is not. AppStat Lab II teaches `TSNE(perplexity=30, init='pca')` and `UMAP(n_neighbors=15, min_dist=0.1)` as standard nonlinear visualisations. Neither is used in PyANNOW. Without them we cannot validate whether Step 2's KMeans labels correspond to *real* clusters or to arbitrary partitions of a smooth distribution. A 2-D t-SNE/UMAP scatter coloured by KMeans label is the simplest possible "is the clustering real?" check.

**Fix plan.**

1. Add `umap-learn` to `PyANNOW/pyproject.toml`'s dependencies (it is in `wormuse-analytics`' deps already).
2. New cell after Step 2:
   ```python
   from sklearn.manifold import TSNE
   import umap

   Xs = StandardScaler().fit_transform(X_neural.T)   # (T, 302)
   Y_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(Xs)
   Y_tsne = TSNE(perplexity=30, init='pca', random_state=0).fit_transform(Xs)

   fig, axes = plt.subplots(1, 2, figsize=(12, 5))
   for ax, Y, name in [(axes[0], Y_umap, 'UMAP'), (axes[1], Y_tsne, 't-SNE')]:
       ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='tab10', s=6, alpha=0.7)
       ax.set(title=f'{name} coloured by Step 2 KMeans label')
   plt.tight_layout(); plt.show()
   ```
3. Markdown note on the t-SNE caveat: inter-cluster distance and cluster size are *not* meaningful in t-SNE; UMAP preserves more global structure.

Reference implementation: `wormuse-analytics/src/wormuse_analytics/dimreduction.py` — `nonlinear_view`.

**Affected files.**
- `PyANNOW/pyproject.toml` — add `umap-learn>=0.5`.
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — new cell after Step 2.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/TODO.md` — this entry.
