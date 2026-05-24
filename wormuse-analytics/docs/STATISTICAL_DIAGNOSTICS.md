# Improving the worm→Chopin pipeline through Applied Statistics

**`wormuse-analytics` — the AppStat-driven engineering doc.**
*Living document. Owned by `[@appstat-audit]`. Updated as each notebook section lands.*

---

## 0. The goal

**Build the best statistical model that maps a *C. elegans* nervous-system / muscle activity into a piano performance resembling Chopin's Nocturne in C♯ minor, Op. posth.**

That goal has two parts:

1. **A pipeline** — ion channels → neurons → muscles → note events → piano audio.
2. **A scoring function** — quantifies "Chopin-likeness" rigorously enough that *optimising it actually produces music*.

PyANNOW's notebook 03 already implements eight pipeline stages, each badged with one NAML lecture. This document **does not throw that out**. It does two things:

- **Audit the pipeline against the AppStat 2026 curriculum** (Lectures 00-07, Labs I-VI) to find where the current implementation fails AppStat best practice.
- **Re-engineer each stage** using the AppStat tool that fits — proper standardization, biplot interpretability, four-method clustering comparison, OLS diagnostics + Lasso for sparsity, calibrated logistic onset detector, multi-tolerance F1 + PR + ROC + bootstrap CIs, and a Random-Forest model-agnostic ceiling.

The deliverables of this project (one md = this doc; one notebook `01_appstat_lecture_audit.ipynb`; one HTML presentation; one Python package `wormuse_analytics`; and 21 new PyANNOW issues) collectively answer: *given the current pipeline, what is the strongest AppStat-correct version we can build, and how does it actually perform?*

---

## 1. The current pipeline (PyANNOW notebook 03)

```
shared/examples/chopin_nocturne_op_posth_csharp_minor.mid
                                            │
                                            ▼  parse_midi
              ┌────────────────────────────────────────────┐
              │      Chopin events (list[NoteEvent])       │
              └────────────────────────────────────────────┘
                                            │
                  ┌─────────────────────────┴───────────────────────┐
                  │                                                 │
            note_onsets                              build_chopin_features(k=8)
                  │                                                 │
                  ▼                                                 ▼
          t_on_chopin (M,)                              C_chopin (T, k_chopin=8)
                                                                    ▲
                                                                    │
ion-channel HH params (DEFAULT_PARAMS)                              │ supervised target
        │                                                           │ for steps 1b-6
        ▼  run_forward_fast                                         │
  V_muscles (T, 8)        ──────────────►  (synthetic 302 neurons) ─┘
        │                                  ──────────────┐
        │                                                │
        ▼                                                ▼
 onsets_base = onsets_from_result               X_neural (302, T)
   (Step 0 baseline)                                     │
                                                         ▼
                                            Step 1a RSVD → U_k, Σ_k, V_k^T (k=4)
                                                         │
                                                         ▼
                                            Step 1b Procrustes → Z_aligned
                                                         │
                                                         ▼
                                            Step 2  PCA + KMeans → cluster labels
                                                         │
                                                         ▼
                                            Step 3  RidgeCV → C_pred
                                                         │
                                                         ▼
                                            Steps 4-6  MLP + Adam + L-BFGS → C_pred
                                                         │
                                                         ▼
                                            Step 8a   ODE-PINN (oscillator)
                                            Step 8b   PDE-PINN (1D wave)
                                                         │
                                                         ▼
                          all activation envelopes → find_peaks(distance=280ms, height=mean)
                                                         │
                                                         ▼
                                                  onsets_step_k
                                                         │
                                                         ▼  pitch ← MUSCLE_PITCHES[idx % 8]
                                                  MIDI events
                                                         │
                                                         ▼  synthesise_melody (piano FM)
                                                       audio
                                                         │
                                                         ▼  onset_loss / musical_f1 vs t_on_chopin
                                                  score per step
```

Each NAML step adds machinery to the **encoder** (left half) and the **mapper** (centre); the **note generator** (right half) is shared across all steps.

---

## 2. Ten structural problems in the current pipeline

These are not metric problems — they are **logic problems**. The metric-only fixes in PyANNOW ISSUE-016 and ISSUE-017 cannot resolve them; they have to be addressed in the pipeline itself.

The numbers reference new PyANNOW issues this audit drafts (ISSUE-029 to ISSUE-038, drafted in `docs/proposed_pyannow_issues/`).

| # | Problem | Where it lives | AppStat tool that fixes it |
|---|---|---|---|
| 1 | **Synthetic 302-neuron matrix** — the "302 neurons" are 8 muscle voltages repeated with noise (`X_neural[i] = V_mus[:, i%8] + N(0, 0.05)`). Any "neural-state compression" therefore trivially recovers 8 blocks. | `_build_naml_progression_nb.py` cell 4 | L01 — PCA biplot reveals it instantly; rank ≈ 8 (or fewer), not 302. **ISSUE-029** |
| 2 | **Step 0 mislabelled "random / no NAML".** It actually uses the deterministic body-wave model `run_forward_fast` with phase-gated crest detection — a biologically structured baseline. Calling it "random notes" misleads readers. | notebook §Step 0 | L00 — descriptive statistics (its IOIs cluster at one body-wave period). **ISSUE-030** |
| 3 | **Pitch bottleneck.** 8 muscle groups → 8 pitches via `MUSCLE_PITCHES = pentatonic`. Chopin uses ~80 distinct pitches. No statistical method can lift this hard cap; it must be addressed by either (a) lifting `n_muscles → 95` everywhere or (b) acknowledging the ceiling in the score. | `composer/worm_optimizer.py:152` | L06 — pitch-aware F1: a worm with the wrong pitch should not get credit. **ISSUE-031** |
| 4 | **Procrustes alignment between incommensurate spaces.** `Z_worm ∈ ℝ^{T×4}` (PCA of noisy-muscle activity) is rotated onto `C_chopin ∈ ℝ^{T×8}` (PCA of a binary piano roll). Different units, different feature semantics → the "rotation" has no physical interpretation. | `step1_svd/procrustes.py` | L04 — standardize features before any linear map; L01 — biplot to show what each space encodes. **ISSUE-032** |
| 5 | **All steps share one peak detector with a magic threshold.** Every step from 1-6 ends with `find_peaks(activ, distance=..., height=activ.mean())`. The step differs only upstream — but the same `mean()` threshold is applied to wildly different activation distributions. | every step | L05 — replace with a calibrated `LogisticRegression(class_weight='balanced')` per step; tune threshold by Youden's J or best-F1. **ISSUE-033** |
| 6 | **Chopin features lossily compressed before training.** `build_chopin_features` does `roll = piano_roll → centre → SVD → keep k=8`. Anything Chopin has in dimensions ≥9 is invisible to the model. | `step1_svd/procrustes.py:75` | L01 — report the cumulative variance captured by k=8 (typically much less than 90%); raise k or fit on the full roll. **ISSUE-034** |
| 7 | **Onset-only metric.** `onset_loss` and `musical_f1` only score *when* notes happen — they don't penalise wrong pitches or wrong dynamics. A worm that gets the rhythm right but plays the wrong notes scores full marks. | `targets/midi_target.py:131` | L06 — pitch-aware F1 with bipartite matching; velocity correlation. **ISSUE-035** |
| 8 | **Step 8 PINN is *not* the ion-channel PINN.** The advertised "PINN centerpiece" (ION_CHANNELS.md) learns Hodgkin-Huxley gating kinetics. The actual Step 8 trains on a **damped oscillator ODE** and a **1-D wave PDE**, with Chopin features as the data target. Two different physics; the connection to the biology is broken. | `step8_pinn/locomotion_pinn.py` | L04 — out of AppStat scope, but L06's RandomForest ceiling (L07) reveals whether the PINN's "physics" buys anything beyond a non-physics ML model. **ISSUE-036** |
| 9 | **"Biological ceiling" calculation overstated.** The note-rate ceiling formula assumes any muscle can fire any pitch ("95 voices → all of Chopin's notes reachable"). In code there is no muscle-to-key mapping that allows this; one muscle ↔ one fixed pitch. | `targets/midi_target.py:biological_ceiling` | L06 — confusion matrix of which Chopin pitches are even *in the reachable set*. **ISSUE-037** |
| 10 | **No cross-validation anywhere.** Every per-step number is in-sample on the same 10s Chopin window. No held-out folds, no temporal CV; sub-window variability is unknown. | every step | L06 — `KFold` over Chopin sub-windows; bootstrap CIs (already covered by ISSUE-021). **ISSUE-038** |

---

## 3. AppStat-driven re-engineering, lecture by lecture

For each AppStat 2026 lecture: what tool it offers, what it fixes in the pipeline above, and what the corresponding notebook section produces.

Legend: ✅ already done by PyANNOW · 🟡 partial · 🔴 missing.

### L00 — Introduction · descriptive statistics → Lab I

**Tool.** Mean / median / std / **skew** / **kurtosis**; histograms, KDE, pair plots, boxplots. The principle: *plot first, then test*.

**What the pipeline needs from L00.**
- Show each step's IOI distribution as a KDE alongside Chopin's. The naked eye then sees that Step 0 is unimodal at ~220 ms while Chopin is broad — *no scalar metric can be trusted to summarise this difference*. (Logic problem #2 surfaces here visually.)
- Compute per-step descriptive statistics (n_notes, ioi_mean, ioi_std, ioi_skew, ioi_kurtosis) into a single DataFrame.

**Status.** 🔴 missing in PyANNOW → **`[@appstat-audit] ISSUE-022`** (already drafted).

**`wormuse_analytics.descriptive`** delivers `step_summary`, `collect_step_stats`, `plot_ioi_distributions`.

### L01 — PCA · linear dim reduction → Lab II

**Tool.** `StandardScaler + PCA`, scree plot, **biplot** (scores + loadings), cumulative variance ≥ 90 % heuristic, Kaiser rule.

**What the pipeline needs from L01.**
- **Biplot of the 302-neuron PCA** — exposes logic problem #1 (rank≈8) and logic problem #4 (incommensurate spaces). The biplot loadings will show the 8-muscle block structure as 8 nearly-collinear arrow bundles. The reader sees there is no 302-D manifold to compress.
- **Cumulative variance of the Chopin feature matrix** — exposes logic problem #6 (k=8 loses information).
- **Consistent standardization** — Step 1a (raw RSVD) and Step 2 (StandardScaler+PCA) differ; document or fix.

**Status.** ✅ PCA + scree done; 🔴 biplot missing → **`[@appstat-audit] ISSUE-023`**.

**`wormuse_analytics.dimreduction`** delivers `pca_with_scree`, `biplot`.

### L02 — Nonlinear dim reduction (t-SNE, UMAP) → Lab II

**Tool.** `TSNE(perplexity=30, init='pca')` for local structure; `UMAP(n_neighbors=15, min_dist=0.1)` for faster, more global-preserving 2-D projection.

**What the pipeline needs from L02.**
- **Visualise the worm's motor-state manifold in 2-D** with both methods, coloured by the KMeans label from Step 2. If KMeans found genuine clusters, they appear as visually separated colours; if the colouring is mixed, the "motor primitives" are arbitrary partitions of a smooth distribution.

**Status.** 🔴 entirely missing → **`[@appstat-audit] ISSUE-024`**.

**`wormuse_analytics.dimreduction.nonlinear_view`**.

### L03 — Clustering → Lab III + IV

**Tool.** Four paradigms:
- KMeans (spherical, fixed k, silhouette);
- Ward hierarchical (linkage, dendrogram, **cophenetic correlation**);
- DBSCAN (density, finds noise as `-1`);
- GMM (probabilistic / soft, BIC for k).
ARI compares any two labelings.

**What the pipeline needs from L03.**
- **Compare all four methods on the Step 2 PCA scores**. KMeans hard assignments are unphysical for biology (motor primitives transition smoothly); GMM's soft probabilities are closer to reality. The dendrogram tells us whether 4 primitives form a *natural* hierarchy or whether the K=4 split is arbitrary. DBSCAN may flag rare worm behaviour as outliers.

**Status.** ✅ KMeans + silhouette done; 🔴 three others missing → **`[@appstat-audit] ISSUE-025`**.

**`wormuse_analytics.clustering.compare_methods`** returns silhouettes + pairwise ARI.

### L04 — Linear models + diagnostics → Lab V

**Tool.** `statsmodels.api.OLS` for inference; the diagnostic suite — residuals-vs-fitted, QQ-plot, **Breusch-Pagan**, **Durbin-Watson**, **VIF**, **Cook's distance**. Regularisation: Ridge, **Lasso** (sparsity / feature selection), Elastic Net. Always **standardize** before regularised regression.

**What the pipeline needs from L04.**
- **Run the Lab V diagnostics on Step 3's Ridge fit.** PyANNOW currently fits Ridge with `RidgeCV` and stops there — no residual diagnostics, no normality check, no influential-point detection.
- **Run Lasso to test whether k=4 PCs are all needed.** Step 3 hardcodes `k_worm = 4` from a 90% cumulative-variance rule. Lasso says how many PCs the model *actually* uses; if it's < 4, the model is over-specified (logic problem #1's downstream effect).
- **Standardize before Ridge** — currently inconsistent with Step 1a's un-standardized RSVD (logic problem #4 again).

**Status.** ✅ Ridge + RidgeCV done; 🔴 diagnostics + Lasso missing → **`[@appstat-audit] ISSUE-026`**.

**`wormuse_analytics.regression.{diagnose_ridge, lasso_path_selection}`**.

### L05 — Logistic regression

**Tool.** `LogisticRegression(C=...)` for prediction; `sm.Logit` for inference. **Threshold tuning** — default 0.5 is rarely optimal; tune by Youden's J (`TPR − FPR`) or best-F1.

**What the pipeline needs from L05.**
- **Replace the shared `find_peaks(height=mean())` onset detector** (logic problem #5) with a calibrated single-feature logistic model on each step's activation envelope. Then tune the threshold per step. This separates two questions PyANNOW currently conflates: *did the activation contain the signal?* and *did we pick the right threshold?*

**Status.** 🔴 missing → **`[@appstat-audit] ISSUE-027`**.

**`wormuse_analytics.classification.logistic_onset_detector`**.

### L06 — Classification metrics + model selection → Lab VI

**Tool.** Confusion matrix; precision / recall / **F1**; **ROC** (TPR vs FPR, AUC); **PR curve** (preferred under heavy imbalance, AUC-PR); stratified k-fold CV; `GridSearchCV` inside a `Pipeline` to prevent leakage; class-imbalance handling.

**What the pipeline needs from L06.**
- **Multi-tolerance F1 sweep** — F1@50ms is one operating point; robust steps stay high across {10, 25, 50, 100, 200, 400} ms tolerances; brittle ones collapse.
- **PR + ROC overlay per step** treating the activation envelope as a soft predictor of the binary "onset bin / no onset" label (one bin per 20 ms). AUC-PR is the right summary because onset/no-onset ratio is ~1:20.
- **Bootstrap CIs per step** + paired-bootstrap test for "Step k F1 > Step 0 F1". Without this, every rank claim in PyANNOW is a point estimate with no error bar.
- **Stratified CV over Chopin sub-windows** for variability estimates (logic problem #10).
- **Pitch-aware F1** — bipartite-match worm notes to Chopin notes within ±50 ms AND with the same pitch class. Plain F1 ignores pitch (logic problem #7).

**Status.** ✅ F1@50ms done (ISSUE-016 / 017); 🔴 curves, CIs, CV, pitch-aware metric all missing → **`[@appstat-audit] ISSUE-018, 019, 020, 021, 035`**.

**`wormuse_analytics.metrics`** delivers `f1_vs_tolerance`, `bootstrap_f1`, `paired_bootstrap_compare`, `pitch_aware_f1`. **`wormuse_analytics.classification.{precision_recall_curve_onsets, roc_curve_onsets}`**.

### L07 — Tree-based methods

**Tool.** Decision trees, Random Forest (bagging + feature subsets, OOB error, **permutation importance** > MDI), Gradient Boosting.

**What the pipeline needs from L07.**
- **Random Forest baseline as model-agnostic ceiling.** The current NAML pipeline goes Linear (Ridge) → MLP (Adam) → L-BFGS → PINN, a deep-learning escalation. AppStat insists on a non-deep tabular baseline first. If RF reaches the same F1 as Steps 4-6 MLP, the deep network is buying nothing.
- **Permutation feature importance** per PC — confirms or refutes the k=4 choice from Step 1a/2 in a model-agnostic way.

**Status.** 🔴 missing → **`[@appstat-audit] ISSUE-028`**.

**`wormuse_analytics.trees.{rf_baseline, rf_predicted_onsets}`**.

---

## 4. The improved pipeline (proposed)

After all 21 issues land:

```
ion-channel PINN ──▶ HH-gated spikes ──▶ 95-muscle activations ──▶ muscle voltages V(t)
                                                                          │
                                                                          ▼   StandardScaler
                                                                  Z = PCA(V, k=k90)
                                                                          │  (k90 from 90% cumvar
                                                                          │   of REAL muscle data,
                                                                          │   not synthetic 302-D)
                                                                          ▼
                                                            ┌─────────────┴──────────────┐
                                                            │                            │
                            Lab V — diagnose                │                            │
                          Ridge (R²+VIF+BP+DW+Cook)         │                            │
                            Lasso for k_eff                 │              Lab VII — RF ceiling
                                                            │              (model-agnostic baseline)
                                                            ▼                            │
                                  RidgeCV → C_pred_ridge    MLP+Adam+L-BFGS → C_pred_mlp │
                                                            │                            │
                                                            ▼                            ▼
                                                       activation envelope        rf.predict
                                                            │                            │
                                                            ├────────┬──────┬────────────┤
                                                                     │      │
                                                              Lab VI — calibrated
                                                              LogisticRegression
                                                              + Youden threshold
                                                                     │
                                                                     ▼
                                                              onsets_step_k
                                                                     │
                                            pitch ← argmax_pitch(C_pred ∘ inverse PCA on roll)
                                                                     │
                                                                     ▼
                                                      MIDI events (pitch + onset + velocity)
                                                                     │
                                                                     ▼
                                            score = pitch_aware_F1 across {10, 25, 50, 100, 200} ms
                                                  + AUC-PR + IOI similarity + velocity correlation
                                                  + bootstrap 95% CI on every number
```

Changes from the current pipeline (in order of impact on the final score):

1. **Drop the synthetic 302-neuron matrix.** PCA/RSVD goes directly on `V_muscles ∈ ℝ^{T × 95}` (or on real C302 spike data once ISSUE-010 lands). The k chosen by the 90 % rule on real data is the *true* dimensionality of the worm's motor manifold, not 4-out-of-8-block-structure. (Closes logic #1.)

2. **Predict pitch as well as onset.** `C_pred` ∈ ℝ^{T × k_chopin} is inverse-transformed through the Chopin PCA into a `(T, n_pitches)` piano-roll prediction; at each detected onset the pitch is `argmax_p C_pred_roll[t, p]`. The scoring becomes pitch-aware F1. (Closes logic #7.)

3. **Calibrated logistic onset detector per step**, replacing `find_peaks(..., height=mean())`. (Closes logic #5.)

4. **Standardize everywhere** before any linear method. (Closes logic #4.)

5. **Cross-validate every claim** with stratified or temporal CV over Chopin sub-windows; report 95 % bootstrap CIs. (Closes logic #10.)

6. **Pitch ceiling honesty** — report the maximum F1 a worm with `n_muscles=8` *could* achieve given a one-muscle-one-pitch map: chopin onsets at pitches outside the pentatonic count against recall and tighten the upper bound. (Closes logic #3 and #9.)

7. **Random-Forest ceiling slide** — RF F1 sets the floor every learned model must beat. (Closes the "is the MLP doing anything?" question.)

8. **Step 0 rebadged "Body-wave deterministic baseline"** — its IOI distribution view makes it obvious why it cannot reach Chopin's rhythmic variability with the current oscillator drive. (Closes logic #2.)

9. **Step 8 PINN explicitly labelled "locomotion oscillator PINN, *not* ion-channel PINN"** — the project documentation already promises an ion-channel PINN at the *centre* of the pipeline (`ION_CHANNELS.md`); Step 8 is a *different* PINN. Either rename or rewire. (Closes logic #8.)

10. **Replace the lossy k=8 Chopin PCA with full-roll fitting** for the steps that can handle it (RF, MLP); keep k=8 only where dimensionality forces it (Ridge with regularisation budget). (Closes logic #6.)

---

## 5. Scoring functions — the final composite

The corrected score is a vector, not a scalar:

| Component | What it measures | Range | Floor for "Chopin-like" |
|---|---|---|---|
| `pitch_aware_f1` @ 50 ms tol | onsets matched in both time AND pitch | [0, 1] | ≥ 0.20 (matching ISSUE-016 ad-hoc threshold) |
| AUC-PR (onset bin classification) | threshold-free quality of the activation | [0, 1] | ≥ 0.30 |
| `ioi_similarity` (already in pyannow) | rhythmic distribution overlap | [0, 1] | ≥ 0.30 |
| `velocity_corr` (Pearson on aligned notes) | dynamics match | [-1, 1] | ≥ 0.20 |
| bootstrap 95 % CI width | uncertainty | s.t. CI excludes Step 0 |

A model is "Chopin-like" only when all five floors are passed. None of them is currently reported in PyANNOW notebook 03.

---

## 6. Twenty-one new PyANNOW issues, all tagged `[@appstat-audit]`

Drafted as one md per issue under `docs/proposed_pyannow_issues/`. Ready to paste into `PyANNOW/TODO.md`.

### Metric / readability (already drafted previously)

| ID | Title | Lecture | Pri |
|---|---|---|---|
| 018 | Builder/notebook desync after ISSUE-017 | L06 (infra) | P1 |
| 019 | Cell 20 left panel still misleads | L06 | P1 |
| 020 | Multi-tolerance F1 + PR + ROC sweep | L06 | P2 |
| 021 | Bootstrap CIs for per-step F1 | L06 | P2 |
| 022 | Descriptive stats of per-step outputs | L00 | P2 |
| 023 | PCA biplot + consistent standardization | L01 | P3 |
| 024 | t-SNE / UMAP of motor-state manifold | L02 | P3 |
| 025 | Compare four clustering methods | L03 | P3 |
| 026 | Lab V diagnostics on Step 3 Ridge | L04 | P2 |
| 027 | Logistic onset detector | L05 | P3 |
| 028 | RandomForest baseline + permutation importance | L07 | P2 |

### Pipeline-logic (new — drafted in this revision)

| ID | Title | Logic # | Pri |
|---|---|---|---|
| 029 | Synthetic 302-neuron matrix masquerades as real data | 1 | P1 |
| 030 | Step 0 mislabelled "random / no NAML" | 2 | P3 |
| 031 | 8-muscle pitch bottleneck blocks Chopin | 3 | P1 |
| 032 | Procrustes alignment between unstandardized incommensurate spaces | 4 | P2 |
| 033 | Identical `find_peaks(height=mean)` detector across all steps | 5 | P1 |
| 034 | Chopin features lossily compressed to k=8 before training | 6 | P2 |
| 035 | Pitch-aware F1 missing — onset-only metric ignores wrong notes | 7 | P1 |
| 036 | Step 8 PINN is oscillator PINN, *not* ion-channel PINN | 8 | P2 |
| 037 | Biological ceiling formula assumes any-muscle-any-pitch | 9 | P2 |
| 038 | No cross-validation anywhere | 10 | P2 |

---

## 7. How this document, the notebook, and the modules cooperate

- This document is the *engineering plan and audit*. It is canonical.
- `notebooks/01_appstat_lecture_audit.ipynb` is the *executable demonstration* — each section runs the AppStat tool against the data, reproduces the bad result that the old pipeline gives, and produces the improved one.
- `src/wormuse_analytics/{descriptive, dimreduction, clustering, regression, classification, trees, metrics, loaders, pipeline}.py` are the *reusable building blocks*. `pipeline.py` (added in this revision) wires the improved 8-step pipeline so a user can `import wormuse_analytics.pipeline as wap; wap.improved(meta).run()` and get the corrected scoring table back.
- `presentation/index.html` is the *audience-facing summary* — 11 slides walking through the diagnosis, the fixes, and the new score.
- `docs/proposed_pyannow_issues/ISSUE-018.md` … `ISSUE-038.md` are *drop-in TODO entries* for PyANNOW.

---

## 8. Where to start (for a reviewer)

1. **Just the diagnosis** → read §2 above, then open `presentation/index.html` slides 1-3.
2. **A grader checking the AppStat curriculum coverage** → §3 (the lecture-by-lecture audit), or open the notebook and run all cells.
3. **A future agent making fixes** → §6 (the issue list); each id has a one-pager.
4. **The actual scoring** → run the notebook's "Final" section; the table prints the corrected progression with bootstrap CIs.

---

## 9. Relationship to existing wormuse documents

- `README.md` and `ARCHITECTURE.md` describe what wormuse *intends* to be. This doc explains where the current implementation deviates from that.
- `ION_CHANNELS.md` and `docs/SCIENTIFIC_FOUNDATION.md` describe the *biophysical* and *piano-physics* models. This doc is about the *statistical model* sitting between them.
- `PyANNOW/docs/PyANNOW_NAML_progression.md` is the NAML-side companion (same pipeline viewed through NAML lectures L06-L27). This doc is its AppStat-side mirror.
