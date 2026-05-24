## ISSUE-025 — Compare four clustering methods on motor primitives `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P3 |
| **Severity** | Completeness — only one of four AppStat-canonical methods used |

**Description.** Step 2 runs `KMeans` on the PCA scores and stops. AppStat 2026 Lab III + IV teach four standard methods — KMeans (spherical, hard, fixed k), Ward (hierarchical, dendrogram + cophenetic), DBSCAN (density-based, finds noise as `-1`), GMM (probabilistic, soft, BIC-selected). Without the comparison we have no evidence the k=4 KMeans labels correspond to real motor primitives versus arbitrary partitions of a smooth distribution. GMM in particular is biologically more defensible: real motor primitives transition gradually, not abruptly.

**Fix plan.** Add `pyannow/step2_clustering/motor_primitives.py` helpers:

```python
def compare_clustering_methods(scores, k=4, random_state=0):
    """Run KMeans / Ward / DBSCAN / GMM on the same scores and report ARI."""
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from scipy.cluster.hierarchy import cophenet, fcluster, linkage
    from scipy.spatial.distance import pdist
    # ... see wormuse-analytics/src/wormuse_analytics/clustering.py for reference
```

In notebook 03, add a cell after Step 2 that calls this helper and:
1. Prints silhouette per method.
2. Prints pairwise ARI (DataFrame).
3. 2x2 scatter of PC1/PC2 coloured by each method's labels.
4. Ward dendrogram with cophenetic correlation in the title.

**Reference implementation.** `wormuse-analytics/src/wormuse_analytics/clustering.py` — `compare_methods`, `ward_labels_and_dendrogram`, `dbscan_labels`, `gmm_labels`, `choose_k_bic`.

**Why GMM matters most.** KMeans assigns each timestep to a single hard cluster. Biologically, the worm transitions smoothly between motor primitives over hundreds of ms — GMM's soft probabilities `P(primitive_k | state)` are the correct representation. Replacing the hard label with the argmax of the GMM probabilities is a one-line change; using the full probability vector as a feature for the composer is a multi-step extension.

**Affected files.**
- `PyANNOW/src/pyannow/step2_clustering/motor_primitives.py` — add `compare_clustering_methods`.
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — new cell after Step 2.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/TODO.md` — this entry.
