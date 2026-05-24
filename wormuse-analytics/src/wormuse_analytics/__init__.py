"""wormuse-analytics — Applied-Statistics improvement of the worm→Chopin pipeline.

Goal
----
Build the best statistical model that maps C. elegans nervous-system /
muscle activity into a piano performance resembling Chopin.  This package
is organised one module per AppStat 2026 lecture — each module both
*diagnoses* and *fixes* one stage of the PyANNOW 8-step pipeline.

Modules
-------
    descriptive   — L00 / Lab I    : per-step descriptive statistics (IOI shapes)
    dimreduction  — L01-L02 / Lab II : PCA biplot + t-SNE + UMAP
    clustering    — L03 / Lab III-IV : KMeans + Ward + DBSCAN + GMM comparison
    regression    — L04 / Lab V    : OLS + diagnostics, Lasso (true k), Elastic Net
    classification— L05-L06 / Lab VI : calibrated logistic onset detector, ROC, PR, AUC
    trees         — L07            : RandomForest baseline, permutation importance
    metrics       — L06 + goal     : F1-vs-tolerance, PR/ROC, pitch-aware F1,
                                     velocity correlation, bootstrap CIs
    loaders       — infra          : re-run PyANNOW steps & cache outputs
    pipeline      — goal-driven    : the improved end-to-end run (composite score)

The 21 PyANNOW issues drafted in `docs/proposed_pyannow_issues/` enumerate every
discrepancy between the current implementation and what AppStat best practice
demands, with concrete patch snippets.

See `docs/STATISTICAL_DIAGNOSTICS.md` for the project goal, the 10 logic
problems uncovered in PyANNOW, and the AppStat lecture-by-lecture fix plan.
"""
from __future__ import annotations
