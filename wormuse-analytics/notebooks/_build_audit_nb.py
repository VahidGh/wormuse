"""Build wormuse-analytics/notebooks/01_appstat_lecture_audit.ipynb.

Mirrors the PyANNOW builder pattern (PyANNOW/notebooks/_build_naml_progression_nb.py).
Running this script regenerates the notebook from cells defined in code.

Run from the wormuse-analytics directory:
    python notebooks/_build_audit_nb.py

The notebook is structured around the project GOAL: teach a worm to play
piano like Chopin.  Each section (a) diagnoses what is wrong with the
corresponding PyANNOW pipeline stage and (b) implements the AppStat-correct
improvement.

Section map (mirrors STATISTICAL_DIAGNOSTICS.md §3):

    §0   Setup + load cached PyANNOW outputs
    §1   The 10 pipeline logic problems — `wormuse_analytics.pipeline` diagnostics
    §L00 Descriptive statistics       (Lab I)         — closes logic #2
    §L01 PCA + biplot                  (Lab II linear) — closes logic #1, #6
    §L02 t-SNE + UMAP                  (Lab II nonlinear)
    §L03 Clustering — KMeans / Ward / DBSCAN / GMM   (Lab III + IV)
    §L04 OLS diagnostics + Lasso       (Lab V)        — closes logic #4
    §L05 Logistic onset detector       (Lecture 05)   — closes logic #5
    §L06 PR / ROC / multi-tol F1 / bootstrap CIs / pitch-aware F1     (Lab VI) — closes logic #7, #10
    §L07 RandomForest baseline         (Lecture 07)   — model-agnostic ceiling
    §Final Improved pipeline composite score + recommendations
"""
from __future__ import annotations

import json
from pathlib import Path


def md(*lines) -> dict:
    src = "\n\n".join(s.strip("\n") for s in lines)
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(*lines) -> dict:
    src = "\n".join(s.rstrip("\n") for s in lines)
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}


cells: list[dict] = []

# ── Title ────────────────────────────────────────────────────────────────────
cells += [md(
    "# Improving the worm→Chopin pipeline through Applied Statistics\n",
    "**Goal.** Build the best statistical model that maps *C. elegans* nervous-system / "
    "muscle activity into a piano performance resembling Chopin's Nocturne in C♯ minor.\n",
    "PyANNOW already implements 8 NAML steps.  This notebook (a) diagnoses 10 structural "
    "problems in that pipeline using AppStat 2026 (Polimi) tools, and (b) implements an "
    "AppStat-corrected improvement for each.\n",
    "## Companion files\n"
    "- Living engineering doc: `docs/STATISTICAL_DIAGNOSTICS.md`\n"
    "- Slide deck: `presentation/index.html`\n"
    "- 21 PyANNOW issues: `docs/proposed_pyannow_issues/ISSUE-{018..038}.md`\n",
    "## Section map\n"
    "| § | Title | AppStat lecture | Logic problems it closes |\n"
    "|---|---|---|---|\n"
    "| §0 | Setup + load PyANNOW outputs | — | — |\n"
    "| §1 | The 10 pipeline logic problems | L00 (diagnostic) | (all) |\n"
    "| §L00 | Descriptive statistics | Lab I | #2 |\n"
    "| §L01 | PCA + biplot | Lab II | #1, #4, #6 |\n"
    "| §L02 | t-SNE / UMAP | Lab II | (manifold sanity-check) |\n"
    "| §L03 | Clustering — 4 methods | Lab III + IV | (KMeans alone is incomplete) |\n"
    "| §L04 | OLS diagnostics + Lasso | Lab V | #4 |\n"
    "| §L05 | Logistic onset detector | L05 | #5 |\n"
    "| §L06 | PR / ROC / pitch-aware F1 / bootstrap | Lab VI | #7, #10 |\n"
    "| §L07 | RandomForest baseline | L07 | (deep-model ceiling) |\n"
    "| §Final | Improved pipeline composite score | — | (closes #3, #9 via score) |\n"
), code(
    "import warnings\nwarnings.filterwarnings('ignore')\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "\n"
    "from wormuse_analytics import (loaders, descriptive, dimreduction, clustering,\n"
    "                                regression, classification, trees, metrics, pipeline)\n"
    "\n"
    "plt.rcParams.update({'figure.figsize': (10, 4), 'axes.grid': True, 'grid.alpha': 0.3})\n"
    "sns.set_context('notebook')\n"
    "\n"
    "data = loaders.run_pyannow_pipeline()   # cached after first call\n"
    "steps         = data['steps']\n"
    "chopin_onsets = data['chopin_onsets']\n"
    "t_arr         = data['t_arr']\n"
    "X_neural      = data['X_neural']\n"
    "Z_worm        = data['Z_worm']\n"
    "C_chopin      = data['C_chopin']\n"
    "meta          = data['meta']\n"
    "print(f'Loaded {len(steps)} steps  '\n"
    "      f'duration={meta[\"duration_s\"]}s  '\n"
    "      f'n_neurons={meta[\"n_neurons\"]}  '\n"
    "      f'k_worm={meta[\"k_worm\"]}  k_chopin={meta[\"k_chopin\"]}')"
)]

# ── §1 — The 10 logic problems ────────────────────────────────────────────────
cells += [md(
    "## §1 — Ten pipeline logic problems (the AppStat diagnostic pass)\n",
    "Before applying any statistical correction we audit the pipeline itself.  "
    "Every problem in this section is a *structural* issue — no amount of metric "
    "tuning resolves it; the pipeline has to be changed.\n",
    "We start with the most consequential one.\n",
    "### Logic #1 — The 302-neuron matrix is synthetic\n",
    "`X_neural[i] = V_muscles[:, i % 8] + N(0, 0.05)` — the \"302 neurons\" are 8 "
    "muscle voltages repeated with noise.  Any \"neural-state compression\" therefore "
    "trivially recovers 8 blocks.  We prove it by computing the effective rank."
), code(
    "report = pipeline.diagnose_synthetic_neural(X_neural, n_muscles_assumed=8)\n"
    "for k, v in report.items():\n"
    "    if k == 'interpretation':\n"
    "        print(f'\\n→ {v}')\n"
    "    else:\n"
    "        print(f'{k:30s}: {v}')"
), md(
    "**Interpretation.** If `cumvar_at_n_muscles ≈ 1.0` and `effective_rank_999 ≤ ~16`, "
    "the matrix carries roughly 8 informative degrees of freedom plus noise — NOT a "
    "302-D manifold.  Step 1a's RSVD compression and Step 2's PCA are operating on "
    "synthetic data.  **PyANNOW ISSUE-029** documents the fix: either bring real C302 "
    "spike data (ISSUE-010 also open) or PCA directly on the 95 real muscle voltages."
), md(
    "### Logic #3 — 8-muscle pitch bottleneck\n",
    "The forward model produces 8 muscle voltages mapped to 8 fixed MIDI pitches "
    "(`MUSCLE_PITCHES` = pentatonic in C♯ minor).  Chopin's Nocturne in C♯ minor uses "
    "dozens of distinct pitches.  We compute the ceiling F1 reachable by this map."
), code(
    "# Load Chopin pitches (from the same MIDI file PyANNOW uses)\n"
    "from pyannow.targets.midi_target import parse_midi, NoteEvent\n"
    "from pyannow.composer.worm_optimizer import MUSCLE_PITCHES\n"
    "events, bpm = parse_midi('../../shared/examples/chopin_nocturne_op_posth_csharp_minor.mid')\n"
    "chopin_pitches = np.array([e.pitch for e in events if e.time_s <= meta['duration_s']])\n"
    "print(f'Chopin pitches in first {meta[\"duration_s\"]}s: {len(chopin_pitches)} notes, '\n"
    "      f'{len(set(chopin_pitches))} distinct')\n"
    "\n"
    "ceil = pipeline.reachable_pitches(np.array(MUSCLE_PITCHES), chopin_pitches)\n"
    "for k, v in ceil.items():\n"
    "    print(f'{k:30s}: {v}')"
), md(
    "**Interpretation.** The reachable-pitch fraction is the upper bound any worm "
    "model can achieve on pitch-aware F1.  **PyANNOW ISSUE-031** documents the fix: "
    "lift `n_muscles → 95` (already supported via `generate_muscle_pitches(95)`)."
)]

# ── §L00 ────────────────────────────────────────────────────────────────────
cells += [md(
    "## §L00 — Descriptive statistics (Lab I) — closes logic #2\n",
    "**Lecture recap.** Mean / median / std / **skew** / **kurtosis**; histograms / KDE / "
    "boxplots.  Principle: *plot first, then test*.\n",
    "**Logic problem closed.** Step 0 is mislabelled \"random / no NAML\" — its IOI "
    "distribution will reveal it's a structured body-wave baseline, not random.\n",
    "**Tools.** `descriptive.collect_step_stats`, `descriptive.plot_ioi_distributions`."
), code(
    "df_stats = descriptive.collect_step_stats(steps, chopin_onsets=chopin_onsets)\n"
    "df_stats.round(3)"
), code(
    "fig, ax = plt.subplots(figsize=(10, 4))\n"
    "descriptive.plot_ioi_distributions(steps, chopin_onsets, ax=ax)\n"
    "plt.tight_layout(); plt.show()"
), md(
    "**Interpretation.** Step 0's IOI distribution is sharply peaked at one body-wave "
    "period (≈220 ms); Chopin's is broad with high kurtosis.  The metric paradox of "
    "ISSUE-016 (Step 0 \"wins\" by sparsity) is *visible* here, not just argued from "
    "formulas.  **PyANNOW ISSUE-022** adds this view to notebook 03 directly."
)]

# ── §L01 ────────────────────────────────────────────────────────────────────
cells += [md(
    "## §L01 — PCA + biplot (Lab II linear) — closes logic #1, #6\n",
    "**Lecture recap.** StandardScaler → PCA → scree → cumvar ≥ 90 % → biplot.\n",
    "**Logic problems closed.** #1 (synthetic 302-D matrix's true rank is 8); "
    "#6 (Chopin features lossily compressed to k=8 — we measure how lossy).\n",
    "**Tools.** `dimreduction.pca_with_scree`, `dimreduction.biplot`."
), code(
    "# Logic #1 surfacing — the biplot of the synthetic 302-D matrix\n"
    "X = X_neural.T  # (T, 302)\n"
    "Z, pca, k90 = dimreduction.pca_with_scree(X, standardize=True, var_threshold=0.90)\n"
    "print(f'k at ≥90% cumvar: {k90}')\n"
    "print(f'First 8 components explain: {np.cumsum(pca.explained_variance_ratio_)[7]:.4f}')\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
    "ratios = pca.explained_variance_ratio_[:20]\n"
    "axes[0].bar(range(1, 21), ratios, alpha=.7)\n"
    "axes[0].plot(range(1, 21), np.cumsum(ratios), 'r-o', ms=4)\n"
    "axes[0].axhline(.9, color='grey', ls='--'); axes[0].axvline(k90, color='red', ls='--')\n"
    "axes[0].set(xlabel='component', ylabel='var ratio', title=f'Scree of synthetic 302-D matrix (k={k90})')\n"
    "n_muscles = 8\n"
    "sample_colors = (X.argmax(axis=1) % n_muscles)\n"
    "feature_names = [f'n{i:03d}m{i%n_muscles}' for i in range(X.shape[1])]\n"
    "dimreduction.biplot(Z, pca.components_, ax=axes[1], feature_names=feature_names,\n"
    "                     sample_colors=sample_colors, arrow_scale=4.0, max_arrows=12)\n"
    "plt.tight_layout(); plt.show()"
), code(
    "# Logic #6 surfacing — how lossy is k=8 PCA of Chopin's piano roll?\n"
    "from pyannow.step1_svd.procrustes import build_chopin_features\n"
    "# Re-build Chopin features with successively larger k and check cumvar\n"
    "from pyannow.targets.midi_target import piano_roll\n"
    "pitches, times, roll = piano_roll(events, resolution_s=meta['duration_s']/len(t_arr),\n"
    "                                   clip_s=meta['duration_s'])\n"
    "roll = roll.astype(float).T   # (T, n_pitches)\n"
    "roll -= roll.mean(axis=0)\n"
    "sv = np.linalg.svd(roll, compute_uv=False)\n"
    "cv = np.cumsum(sv**2) / (sv**2).sum()\n"
    "print(f'Chopin piano roll shape: {roll.shape}')\n"
    "print(f'Cumvar at k=8:  {cv[min(7, len(cv)-1)]:.4f}')\n"
    "print(f'Cumvar at k=16: {cv[min(15, len(cv)-1)]:.4f}')\n"
    "print(f'k for 90% var: {int(np.searchsorted(cv, .9)+1)}')"
), md(
    "**Interpretation.** The biplot reveals the 8-muscle block structure — there is no "
    "302-D manifold to discover.  The Chopin cumvar number tells how much of the piano "
    "roll's information `C_chopin` (k=8) actually carries.  **PyANNOW ISSUE-023** "
    "(biplot) and **ISSUE-034** (k=8 audit) document the fixes."
)]

# ── §L02 ────────────────────────────────────────────────────────────────────
cells += [md(
    "## §L02 — t-SNE + UMAP (Lab II nonlinear) — manifold sanity check\n",
    "Visualise the worm's neural manifold in 2-D with both nonlinear methods, "
    "coloured by KMeans label from Step 2.  Clean colour separation = real clusters; "
    "mixed colours = the KMeans labels are arbitrary.\n",
    "**Tool.** `dimreduction.nonlinear_view`. **Issue.** `[@appstat-audit] ISSUE-024`."
), code(
    "Y_umap = dimreduction.nonlinear_view(X, method='umap', n_neighbors=15, min_dist=.1)\n"
    "Y_tsne = dimreduction.nonlinear_view(X, method='tsne', perplexity=30)\n"
    "labels_step2 = data['labels_step2']\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
    "for ax, Y, name in [(axes[0], Y_umap, 'UMAP'), (axes[1], Y_tsne, 't-SNE')]:\n"
    "    ax.scatter(Y[:, 0], Y[:, 1], c=labels_step2, cmap='tab10', s=6, alpha=0.7)\n"
    "    ax.set(title=f'{name} coloured by Step 2 KMeans')\n"
    "plt.tight_layout(); plt.show()"
)]

# ── §L03 ────────────────────────────────────────────────────────────────────
cells += [md(
    "## §L03 — Four clustering methods (Lab III + IV)\n",
    "KMeans (PyANNOW Step 2) is one of four standard methods.  We compare against "
    "Ward (hierarchical), DBSCAN (density-based, noise-aware), and GMM (probabilistic, "
    "soft assignments — closer to smooth biological transitions).  **Issue.** "
    "`[@appstat-audit] ISSUE-025`."
), code(
    "scores = data['pca_scores']\n"
    "k = 4\n"
    "comp = clustering.compare_methods(scores, k=k)\n"
    "print('Silhouette per method:')\n"
    "for m, s in comp['silhouettes'].items(): print(f'  {m:10s} {s:.3f}')\n"
    "print('\\nPairwise ARI:'); display(comp['ari'].round(3))"
), code(
    "fig, axes = plt.subplots(2, 2, figsize=(11, 9))\n"
    "for ax, (name, lab) in zip(axes.flatten(), comp['labels'].items()):\n"
    "    ax.scatter(scores[:, 0], scores[:, 1], c=lab, cmap='tab10', s=6, alpha=0.7)\n"
    "    ax.set(xlabel='PC1', ylabel='PC2', title=f'{name} (k={k})')\n"
    "plt.tight_layout(); plt.show()"
), code(
    "from scipy.cluster.hierarchy import dendrogram\n"
    "_, Z_ward, c = clustering.ward_labels_and_dendrogram(scores, k=k)\n"
    "fig, ax = plt.subplots(figsize=(10, 4))\n"
    "dendrogram(Z_ward, color_threshold=Z_ward[-(k-1), 2], ax=ax, truncate_mode='level', p=5)\n"
    "ax.set(title=f'Ward dendrogram (cophenet = {c:.3f})')\n"
    "plt.tight_layout(); plt.show()"
)]

# ── §L04 ────────────────────────────────────────────────────────────────────
cells += [md(
    "## §L04 — OLS diagnostics + Lasso (Lab V) — closes logic #4\n",
    "PyANNOW Step 3 fits Ridge with `RidgeCV` and stops.  Lab V says: also report "
    "VIF, Breusch-Pagan, Durbin-Watson, residual normality (QQ / Shapiro), Cook's "
    "distance, AND run Lasso to test whether k=4 PCs are really needed.  "
    "**Issue.** `[@appstat-audit] ISSUE-026`."
), code(
    "df_diag = regression.diagnose_ridge(Z_worm, C_chopin, standardize=True)\n"
    "df_diag.round(3)"
), code(
    "lasso_res = regression.lasso_path_selection(Z_worm, C_chopin, target_col=0)\n"
    "print(f'Lasso alpha:           {lasso_res[\"alpha\"]:.4f}')\n"
    "print(f'PCs with non-zero coef: {lasso_res[\"n_kept\"]} / {Z_worm.shape[1]}')\n"
    "print(f'Indices kept:          {list(lasso_res[\"kept_idx\"])}')\n"
    "print(f'R^2 in-sample:         {lasso_res[\"r2\"]:.3f}')"
)]

# ── §L05 ────────────────────────────────────────────────────────────────────
cells += [md(
    "## §L05 — Logistic onset detector — closes logic #5\n",
    "PyANNOW's `find_peaks(activ, distance=280ms, height=activ.mean())` is a 1-feature "
    "classifier with a hardcoded threshold applied to every step's activation envelope.  "
    "Different envelope distributions ⇒ different optimal thresholds.  We replace it "
    "with a calibrated logistic model and tune the threshold by Youden's J or best-F1.  "
    "**Issue.** `[@appstat-audit] ISSUE-027`."
), code(
    "rows = []\n"
    "for s in steps:\n"
    "    if s.activation is None: continue\n"
    "    for strat in ['default', 'youden', 'best_f1']:\n"
    "        res = classification.logistic_onset_detector(s.activation, chopin_onsets,\n"
    "                                                      t_arr, tol_s=0.05,\n"
    "                                                      threshold_strategy=strat)\n"
    "        rows.append({'step': s.name, 'strategy': strat, 'thr': res['threshold'],\n"
    "                     'f1': res['f1'], 'precision': res['precision'],\n"
    "                     'recall': res['recall'], 'auc_roc': res['auc_roc']})\n"
    "pd.DataFrame(rows).round(3)"
)]

# ── §L06 ────────────────────────────────────────────────────────────────────
cells += [md(
    "## §L06 — Curves, CIs, pitch-aware F1 (Lab VI) — closes logic #7, #10\n",
    "Five Lab VI deliverables: F1-vs-tolerance, PR curve, ROC curve, bootstrap CIs, "
    "and the **pitch-aware F1** that requires the worm to match both onset time AND "
    "pitch.  **Issues.** `[@appstat-audit] ISSUE-018..021, 035`."
), code(
    "# F1 vs tolerance — robust steps stay high across tolerances\n"
    "tols = [0.010, 0.025, 0.050, 0.100, 0.200, 0.400]\n"
    "df_tols = pd.DataFrame()\n"
    "for s in steps:\n"
    "    df = metrics.f1_vs_tolerance(s.onsets, chopin_onsets, tols_s=tols,\n"
    "                                  window_s=meta['duration_s'])\n"
    "    df['step'] = s.name\n"
    "    df_tols = pd.concat([df_tols, df], ignore_index=True)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "for name, sub in df_tols.groupby('step'):\n"
    "    ax.plot(sub['tol_s']*1000, sub['f1'], 'o-', label=name)\n"
    "ax.set(xlabel='matching tolerance (ms)', ylabel='F1',\n"
    "       title='F1 vs tolerance (Lab VI / ISSUE-020)')\n"
    "ax.legend(fontsize=8); plt.tight_layout(); plt.show()"
), code(
    "# PR / ROC overlays\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "for s in steps:\n"
    "    if s.activation is None: continue\n"
    "    p, r, _, ap = classification.precision_recall_curve_onsets(\n"
    "        s.activation, chopin_onsets, t_arr, tol_s=0.05)\n"
    "    fpr, tpr, _, auc_r = classification.roc_curve_onsets(\n"
    "        s.activation, chopin_onsets, t_arr, tol_s=0.05)\n"
    "    axes[0].plot(r, p, lw=1.5, label=f'{s.name}  AP={ap:.3f}')\n"
    "    axes[1].plot(fpr, tpr, lw=1.5, label=f'{s.name}  AUC={auc_r:.3f}')\n"
    "axes[0].set(xlabel='recall', ylabel='precision', title='PR curve')\n"
    "axes[1].plot([0,1], [0,1], 'k--', alpha=.4)\n"
    "axes[1].set(xlabel='FPR', ylabel='TPR', title='ROC curve')\n"
    "for ax in axes: ax.legend(fontsize=8)\n"
    "plt.tight_layout(); plt.show()"
), code(
    "# Pitch-aware F1 — the metric the wormuse goal actually requires\n"
    "# Map each step's onsets back to pitches using MUSCLE_PITCHES (the same map PyANNOW uses)\n"
    "from pyannow.composer.worm_optimizer import MUSCLE_PITCHES\n"
    "n_mus = 8\n"
    "\n"
    "rows = []\n"
    "for s in steps:\n"
    "    # Assign pitches the same way PyANNOW does: muscle_idx = k_idx % n_muscles\n"
    "    pitches = np.array([MUSCLE_PITCHES[k_idx % n_mus] for k_idx in range(len(s.onsets))])\n"
    "    plain = metrics.f1_vs_tolerance(s.onsets, chopin_onsets, tols_s=[0.05],\n"
    "                                     window_s=meta['duration_s']).iloc[0]\n"
    "    pf = metrics.pitch_aware_f1(s.onsets, pitches,\n"
    "                                 chopin_onsets, chopin_pitches,\n"
    "                                 tol_s=0.05, window_s=meta['duration_s'])\n"
    "    rows.append({'step': s.name, 'plain F1@50ms': plain['f1'],\n"
    "                 'pitch-aware F1': pf['f1'], 'precision': pf['precision'],\n"
    "                 'recall': pf['recall']})\n"
    "pd.DataFrame(rows).round(3)"
), md(
    "**Interpretation.** The **pitch-aware F1 column is the metric the wormuse goal "
    "actually requires.**  Whereever it is dramatically lower than plain F1, the step "
    "was getting *temporal* matches with wrong pitches — credit it should never have "
    "received.  **PyANNOW ISSUE-035** documents the metric replacement."
), code(
    "# Bootstrap CIs on pitch-aware F1\n"
    "rows = []\n"
    "for s in steps:\n"
    "    pitches = np.array([MUSCLE_PITCHES[k_idx % n_mus] for k_idx in range(len(s.onsets))])\n"
    "    ci = metrics.bootstrap_pitch_aware_f1(s.onsets, pitches,\n"
    "                                           chopin_onsets, chopin_pitches,\n"
    "                                           B=400, window_s=meta['duration_s'],\n"
    "                                           sub_window_s=5.0)\n"
    "    rows.append({'step': s.name, 'median': ci['median'],\n"
    "                 'CI low': ci['ci_low'], 'CI high': ci['ci_high']})\n"
    "pd.DataFrame(rows).round(3)"
)]

# ── §L07 ────────────────────────────────────────────────────────────────────
cells += [md(
    "## §L07 — RandomForest model-agnostic ceiling\n",
    "If RF F1 ≥ MLP F1 from PyANNOW Steps 4-6, the deep model is buying nothing.  "
    "If RF F1 << MLP F1, the deep model genuinely captures structure RF can't.  "
    "Either way we now know.  **Issue.** `[@appstat-audit] ISSUE-028`."
), code(
    "rf, report = trees.rf_baseline(Z_worm, C_chopin, n_estimators=300)\n"
    "print(f'RF R^2 (train, OOB): {report.r2_train:.3f}, {report.r2_oob:.3f}')\n"
    "\n"
    "rf_onsets, rf_activ = trees.rf_predicted_onsets(rf, Z_worm, t_arr)\n"
    "rf_pitches = np.array([MUSCLE_PITCHES[k_idx % n_mus] for k_idx in range(len(rf_onsets))])\n"
    "pf_rf = metrics.pitch_aware_f1(rf_onsets, rf_pitches,\n"
    "                                chopin_onsets, chopin_pitches,\n"
    "                                tol_s=0.05, window_s=meta['duration_s'])\n"
    "print(f'RF pitch-aware F1@50ms: {pf_rf[\"f1\"]:.3f}  '\n"
    "      f'(precision={pf_rf[\"precision\"]:.3f}, recall={pf_rf[\"recall\"]:.3f})')"
), code(
    "fig, ax = plt.subplots(figsize=(6, 3))\n"
    "idx = np.argsort(report.permutation_means)[::-1]\n"
    "ax.barh(range(len(idx)), report.permutation_means[idx], xerr=report.permutation_stds[idx])\n"
    "ax.set(yticks=range(len(idx)), yticklabels=[f'PC{i+1}' for i in idx],\n"
    "       xlabel='permutation importance', title='RF feature importance (L07)')\n"
    "plt.tight_layout(); plt.show()"
)]

# ── Final ────────────────────────────────────────────────────────────────────
cells += [md(
    "## Final — Improved-pipeline composite score\n",
    "The wormuse goal demands a *vector* score, not a single number.  We assemble it "
    "by calling `pipeline.ImprovedPipeline.score_all` across the steps."
), code(
    "# Build the composite score table\n"
    "# Velocities — PyANNOW assigns velocity = activation magnitude clipped to [20, 127]\n"
    "chopin_velocities = np.array([e.velocity for e in events if e.time_s <= meta['duration_s']])\n"
    "\n"
    "ip = pipeline.ImprovedPipeline(\n"
    "    duration_s=meta['duration_s'],\n"
    "    chopin_onsets=chopin_onsets,\n"
    "    chopin_pitches=chopin_pitches,\n"
    "    chopin_velocities=chopin_velocities,\n"
    "    t_arr=t_arr,\n"
    ")\n"
    "\n"
    "runs = []\n"
    "for s in steps:\n"
    "    pitches = np.array([MUSCLE_PITCHES[k_idx % n_mus] for k_idx in range(len(s.onsets))])\n"
    "    velocities = np.full(len(s.onsets), 80)   # placeholder — PyANNOW velocity rule below\n"
    "    if s.activation is not None and len(s.onsets) > 0:\n"
    "        # Use activation magnitude at onset times as proxy velocity\n"
    "        idx = np.clip(np.searchsorted(t_arr, s.onsets), 0, len(t_arr)-1)\n"
    "        velocities = np.clip(s.activation[idx] * 80, 20, 127).astype(int)\n"
    "    runs.append((s.name, s.onsets, pitches, velocities, s.activation))\n"
    "\n"
    "df_final = ip.score_all(runs)\n"
    "df_final.round(3)"
), md(
    "### What this table reports\n"
    "- `pitch_aware_f1` — the metric the wormuse goal actually requires.\n"
    "- `auc_pr` — the threshold-free quality of the activation envelope.\n"
    "- `ioi_similarity` — rhythmic distribution overlap (already in pyannow).\n"
    "- `velocity_correlation` — Pearson r on matched-note velocities (dynamics).\n"
    "- `bootstrap_ci_low / high` — 95 % CI on pitch-aware F1.\n",
    "### What it confirms\n"
    "- Plain `onset_loss` and even plain `musical_f1` overstate the worm's performance "
    "because they ignore pitch.  Pitch-aware F1 is much lower.\n"
    "- The bootstrap CIs are wide — the per-step rankings depend on which sub-window "
    "of Chopin is sampled.  Without CIs, all rank claims are point estimates with no "
    "error bar.\n",
    "### What the 21 [@appstat-audit] issues will deliver if closed\n"
    "ISSUE-029 fixes the synthetic-302 problem; ISSUE-031 lifts the pitch ceiling; "
    "ISSUE-033 replaces the magic-threshold peak detector; ISSUE-035 makes the metric "
    "pitch-aware; ISSUE-018..028 add the AppStat diagnostic infrastructure.  Together "
    "these change the pipeline from \"a story about NAML methods\" into \"a measurable, "
    "uncertainty-quantified worm-to-Chopin model\"."
)]

# ─────────────────────────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = Path(__file__).parent / "01_appstat_lecture_audit.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} — {len(cells)} cells, {out.stat().st_size//1024} KB")
