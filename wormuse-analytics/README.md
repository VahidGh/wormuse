# wormuse-analytics

**The AppStat sub-project.** Python notebooks that analyze the worm-music data: descriptive stats, PCA, clustering, regression with diagnostics, classification. Maps 1:1 to the current 2026 Python-based AppStat course (lectures 00-07, labs I-VI).

---

## Layout

```
wormuse-analytics/
├── pyproject.toml
├── src/wormuse_analytics/
│   ├── loaders.py            Load WCON pose + spike JSON + MIDI from shared/examples/
│   ├── metrics.py            Music-quality scoring functions
│   ├── plots.py              Reusable plot helpers matching AppStat lab style
│   └── __init__.py
├── notebooks/
│   ├── Lab_I_descriptive.ipynb       Descriptive stats (Lab I pattern)
│   ├── Lab_II_PCA.ipynb              PCA + t-SNE + UMAP on 302-D states (Lab II)
│   ├── Lab_III_clustering.ipynb      KMeans + Ward + silhouette on motor primitives (Lab III)
│   ├── Lab_V_regression.ipynb        OLS + diagnostics, music-quality vs ion-channel params (Lab V)
│   └── Lab_VI_classification.ipynb   Logistic + ROC + RF for musical/noise (Lab VI)
└── data/                     Small sample CSVs (the big ones live in shared/examples/)
```

The notebook names deliberately mirror the AppStat 2026 lab names — when graders / classmates open this folder it looks immediately familiar.

## Stack

```bash
pip install numpy pandas matplotlib seaborn scipy \
            scikit-learn statsmodels \
            umap-learn tqdm \
            jupyter
```

Mirrors the imports of the AppStat labs (Lab II → Lab VI) extracted in the `polimi-appstat` skill.

## Lecture map (AppStat 2026)

### Lab I — Descriptive statistics (Lecture 00)

| Concept | Use in this project |
|---|---|
| Mean, median, std | Per-neuron firing rate, inter-spike interval distribution |
| Skewness, kurtosis (`scipy.stats.skew`, `kurtosis`) | Are spike rates Gaussian or heavy-tailed? |
| Box plots, histograms | Distributions per motor neuron group |
| Pair plots (seaborn) | Cross-neuron correlations |

### Lab II — PCA / nonlinear (Lectures 01-02)

| Concept | Use in this project |
|---|---|
| `StandardScaler` + `PCA` | Reduce 302-D neural trajectories to 2-10 D |
| Scree plot, cumulative variance ≥ 90% | Choose `k` |
| Biplot (loadings + scores) | Identify which neurons dominate PC1, PC2 |
| t-SNE, UMAP | Nonlinear projection; identify motor-state manifold structure |

### Lab III — Clustering (Lecture 03)

| Concept | Use in this project |
|---|---|
| KMeans + silhouette + elbow | Find optimal number of motor primitives |
| `scipy.cluster.hierarchy.linkage` (Ward), `dendrogram` | Hierarchical cluster of motor states |
| DBSCAN | Distinguish "rare" worm behaviors |
| GMM | Probabilistic cluster assignments for fuzzy transitions |
| `adjusted_rand_score` | Compare KMeans vs Ward vs GMM agreement |

### Lab V — Linear models + diagnostics (Lecture 04)

| Concept | Use in this project |
|---|---|
| `statsmodels.api.OLS` + `summary()` | Regress music quality on `(τ_m, τ_h, V_thresh, g_K, …)` |
| Residuals vs fitted, QQ plot | Check linearity + normality |
| `het_breuschpagan` (BP test) | Heteroskedasticity check |
| `durbin_watson` | Autocorrelation in time-ordered data |
| `variance_inflation_factor` (VIF) | Multicollinearity among ion-channel params |
| `cooks_distance` | Influential scenarios |
| Ridge / Lasso / Elastic Net | Regularization for the high-VIF case |

### Lab VI — Classification + ROC (Lectures 05-06)

| Concept | Use in this project |
|---|---|
| `LogisticRegression` (binary) | "Musical" vs "noisy" generated melodies (labels from human or heuristic rater) |
| `confusion_matrix`, `classification_report` | Per-class performance |
| `roc_curve`, `roc_auc_score` | Threshold-free quality of the classifier |
| `cross_val_score`, `StratifiedKFold` | Validate generalization |
| Decision Tree, Random Forest (Lecture 07) | Stronger baseline; feature importance |

## Phases

- **Phase 4** — All five notebooks running on a ≥ 50-scenario dataset produced by Phases 1-3.
- **Phase 6** — Notebooks feed back into PyANNOW training (use the regression diagnostics to find which ion-channel params actually matter; drop the rest).
- **Phase 8** — Polish each notebook into a thesis-quality exhibit.

See [../ROADMAP.md](../ROADMAP.md) for the full plan.

## Conventions

- Each notebook starts with a **markdown header** stating the lab analogue and the lecture concept it implements.
- Cell structure: **load → preprocess → analyze → plot → interpret**. Every analysis ends with a markdown cell saying *what the result means* in plain English.
- Standardize before PCA / clustering / distance-based methods (`StandardScaler`).
- `random_state=0` everywhere reproducibility matters.
- Use `Pipeline([("scale", StandardScaler()), ("clf", ...)])` for any cross-validated workflow — prevents leakage of test data into preprocessing.

## When stuck

- `polimi-appstat` skill: `dimensionality-reduction.md`, `clustering.md`, `linear-models.md`, `classification.md`, `tree-methods.md`, `exam-patterns.md`.
- The 2 working snippets (clustering-pipeline + regression-with-diagnostics) are immediate templates.
- For statistical hypothesis testing depth beyond the labs: `data:statistical-analysis` skill.
- For analysis quality-checking before sharing: `data:validate-data` skill.
- For deeper linear-algebra theory behind the methods: `polimi-naml` (svd, pca) and `polimi-nla`.
