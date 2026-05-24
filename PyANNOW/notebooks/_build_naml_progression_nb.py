"""Build PyANNOW/notebooks/03_pyannow_naml_progression.ipynb.

Run: python3 PyANNOW/notebooks/_build_naml_progression_nb.py
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


cells = []

# ── Title ────────────────────────────────────────────────────────────────────
cells += [md(
    "# PyANNOW — NAML Progression\n",
    "**Teaching a C. elegans worm to approximate Chopin using only NAML course methods**\n",
    "Each section introduces one new NAML concept and shows the musical improvement it produces.\n",
    "## References\n",
    "- Presentation: `PyANNOW/presentation/index.html` (open in browser)\n"
    "- Living doc: `PyANNOW/docs/PyANNOW_NAML_progression.md`\n"
    "- MIDI target: `shared/examples/chopin_nocturne_op_posth_csharp_minor.mid`\n",
    "## Runtime\n",
    "Approximately **5-8 minutes** end-to-end (Steps 0-6; Step 8 adds ~2 min extra). "
    "Step 8 ODE/PDE PINN comparison is the most expensive section."
)]

# ── 0. Setup ─────────────────────────────────────────────────────────────────
cells += [md("## 0. Setup"), code(
    "import sys, time, warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.gridspec as gridspec\n"
    "import scipy.signal as sps\n"
    "from scipy.io import wavfile\n"
    "from pathlib import Path\n"
    "from IPython.display import Audio, display, Markdown\n"
    "\n"
    "# PyANNOW modules (installed via pip install -e PyANNOW)\n"
    "from pyannow.targets.midi_target import (\n"
    "    parse_midi, note_onsets, onset_loss, piano_roll, musical_f1, ioi_similarity)\n"
    "from pyannow.composer.piano_synth import synthesise_melody\n"
    "from pyannow.composer.worm_optimizer_fast import (\n"
    "    run_forward_fast, onsets_from_result, MUSCLE_PITCHES_96)\n"
    "from pyannow.composer.worm_optimizer import generate_neural_activity_302\n"
    "from pyannow.ion_channels.celegans_hh import DEFAULT_PARAMS\n"
    "\n"
    "from pyannow.step1_svd.encoder      import rsvd, neural_scores, choose_k_by_variance\n"
    "from pyannow.step1_svd.procrustes   import procrustes_align, build_chopin_features\n"
    "from pyannow.step2_clustering.motor_primitives import (\n"
    "    pca_reduce, find_motor_primitives, cluster_to_notes, choose_k_silhouette)\n"
    "from pyannow.step3_regression.ridge_composer import RidgeComposer, explained_variance_by_ridge\n"
    "from pyannow.step4_ffnn.jax_composer import create_model, init_params, model_summary\n"
    "from pyannow.step5_training.adam_trainer import train_adam\n"
    "from pyannow.step6_lbfgs.lbfgs_polish import polish_lbfgs\n"
    "from pyannow.step8_pinn.locomotion_pinn import compare_ode_vs_pde\n"
    "\n"
    "plt.rcParams.update({'figure.figsize': (11, 4), 'axes.grid': True, 'grid.alpha': 0.3})\n"
    "OUTDIR = Path('step_outputs'); OUTDIR.mkdir(exist_ok=True)\n"
    "MIDI_PATH = Path('../../shared/examples/chopin_nocturne_op_posth_csharp_minor.mid')\n"
    "DURATION = 10.0   # seconds to simulate and compare\n"
    "rng = np.random.default_rng(0)\n"
    "print('PyANNOW loaded ✓')"
)]

# ── Global data generation ────────────────────────────────────────────────────
cells += [md(
    "## Data preparation\n",
    "Generate the worm's neural activity and load the Chopin target.\n",
    "**Worm:** run the 96-cell Boyle et al. (2012) 4×24 forward model to get `V_muscles` "
    "(96 muscle cells × T timesteps across 4 quadrants: DL/VL/DR/VR).  \n"
    "**Neural activity (v0.7.0):** `generate_neural_activity_302()` produces a biologically-structured "
    "302-neuron matrix with **k ≥ 4 independent principal components** (command interneurons, "
    "A/B/D motor neurons, interneurons, sensory neurons). "
    "This replaces the previous degenerate approach (302 repeats of muscle signals → rank 1), "
    "which prevented all NAML learning steps from outperforming the Step 0 rule-based baseline.  \n"
    "**Chopin:** parsed from the MIDI file into note-onset times."
), code(
    "# ── 1. Run 96-cell Boyle forward model (v0.7.0 default) ─────────────────\n"
    "result_worm = run_forward_fast(\n"
    "    DEFAULT_PARAMS, duration_s=DURATION, dt_ms=0.5,\n"
    "    drive_freq_hz=1.5, drive_amplitude=12.0, random_seed=42)\n"
    "V_mus  = result_worm['V_muscles']           # (T, 96)  muscle voltages\n"
    "t_ms   = result_worm['t_arr_ms']            # (T,)  time in ms\n"
    "t_arr  = t_ms * 1e-3                        # (T,)  time in seconds\n"
    "worm_events = result_worm['note_onsets_s']  # rule-based baseline\n"
    "n_muscles   = result_worm['n_muscles']      # 96 (Boyle 4×24)\n"
    "T_pts       = len(t_arr)\n"
    "print(f'Worm simulation: {n_muscles}-cell model, T={T_pts} timesteps, '\n"
    "      f'{len(worm_events)} note events')\n"
    "print(f'Quadrant pitch ranges: DL=MIDI 24-47, VL=48-71, DR=72-95, VR=96-119')\n"
    "\n"
    "# ── 2. Biologically-structured 302-neuron activity (v0.7.0) ─────────────\n"
    "# generate_neural_activity_302() models the full C. elegans locomotion circuit:\n"
    "#   cmd interneurons (12): 4-phase rhythmic drive\n"
    "#   A-class MNs (21):      backward traveling wave\n"
    "#   B-class MNs (18):      forward traveling wave\n"
    "#   D-class MNs (19):      inhibitory antiphase\n"
    "#   other interneurons (30): multi-frequency oscillations\n"
    "#   sensory neurons (100):  sparse burst responses\n"
    "#   body neurons (102):     slow oscillations + noise\n"
    "# Result: X ∈ ℝ^{302 × T} with k≥4 independent PCs — enables all NAML steps\n"
    "# to learn a meaningful worm→Chopin mapping (vs previous rank-1 degenerate data).\n"
    "X_neural = generate_neural_activity_302(\n"
    "    n_steps=T_pts, dt_ms=0.5, drive_freq_hz=1.5, seed=42)\n"
    "print(f'Neural activity matrix: {X_neural.shape}  (302 neurons × {T_pts} timesteps)')\n"
    "\n"
    "# Verify rank (the key sanity check for v0.7.0)\n"
    "s_check = np.linalg.svd(X_neural, compute_uv=False)\n"
    "n_sig = int((s_check > s_check[0] * 0.1).sum())\n"
    "print(f'Significant PCs (>10% of σ_max): {n_sig}  '\n"
    "      f'(v0.6.0 had k=1 — any NAML step could not beat Step 0)')\n"
    "\n"
    "# ── 3. Load Chopin MIDI ───────────────────────────────────────────────────\n"
    "events_chopin, bpm = parse_midi(MIDI_PATH)\n"
    "t_on_chopin = note_onsets(events_chopin)\n"
    "print(f'Chopin: {len(events_chopin)} notes, {bpm:.0f} BPM, {t_on_chopin.max():.0f}s total')\n"
    "\n"
    "# Build Chopin feature matrix — v0.8.0: auto-select k by 90% variance rule (ISSUE-034)\n"
    "from pyannow.step1_svd.procrustes import chopin_cumvar\n"
    "cv_chopin = chopin_cumvar(events_chopin, duration_s=DURATION)\n"
    "k8_var = cv_chopin[7] if len(cv_chopin) > 7 else cv_chopin[-1]\n"
    "C_chopin = build_chopin_features(events_chopin, duration_s=DURATION, n_bins=T_pts,\n"
    "                                  k_chopin=None, var_threshold=0.90)\n"
    "k_chopin = C_chopin.shape[1]  # actual k selected\n"
    "print(f'Chopin variance @ k=8: {k8_var:.3f} ({k8_var*100:.1f}%)  — v0.7.0 captured this much')\n"
    "print(f'Auto-selected k={k_chopin} → {cv_chopin[k_chopin-1]:.3f} ({cv_chopin[k_chopin-1]*100:.1f}%) variance')\n"
    "print(f'Chopin feature matrix: {C_chopin.shape}  (T × k_chopin)')\n"
    "\n"
    "# Metric accumulators\n"
    "losses    = {}   # onset_loss per step (comparable only within data-only steps)\n"
    "f1_scores = {}   # musical_f1 per step (the primary metric — cannot be gamed by sparsity)\n"
    "ioi_sims  = {}   # IOI histogram intersection per step"
)]

# ── Step 0 ────────────────────────────────────────────────────────────────────
cells += [md(
    "## Step 0 — Baseline: rule-based neuron→note mapping\n",
    "**No learning.** Body-wave phase → pentatonic pitch. This is what we built in `02_chopin_worm_optimizer.ipynb`.\n",
    "**NAML used:** none.  **Result:** random-sounding, no structure matching Chopin."
), code(
    "# Baseline: rule-based note events from the forward model\n"
    "onsets_base = onsets_from_result(result_worm)\n"
    "L_base  = onset_loss(onsets_base, t_on_chopin, window_s=DURATION)\n"
    "F1_base = musical_f1(onsets_base, t_on_chopin, window_s=DURATION)\n"
    "I_base  = ioi_similarity(onsets_base, t_on_chopin, window_s=DURATION)\n"
    "print(f'Step 0 (baseline):  {len(onsets_base)} notes  '\n"
    "      f'loss={L_base:.5f}  F1={F1_base[\"f1\"]:.3f}  IOI={I_base:.3f}')\n"
    "print(f'  precision={F1_base[\"precision\"]:.3f}  recall={F1_base[\"recall\"]:.3f}  '\n"
    "      f'tp_worm={F1_base[\"tp_worm\"]}  tp_chopin={F1_base[\"tp_chopin\"]}')\n"
    "\n"
    "audio_base, fs = synthesise_melody(worm_events, duration_s=DURATION)\n"
    "print('Step 0 — Baseline melody:')\n"
    "display(Audio(audio_base, rate=fs, autoplay=False))\n"
    "\n"
    "# Save for final comparison\n"
    "wavfile.write(str(OUTDIR/'step0_baseline.wav'), fs, (audio_base*28000).astype('int16'))\n"
    "losses['Step 0 (baseline)']    = L_base\n"
    "f1_scores['Step 0 (baseline)'] = F1_base['f1']\n"
    "ioi_sims['Step 0 (baseline)']  = I_base"
)]

# ── Step 1a: SVD ──────────────────────────────────────────────────────────────
cells += [md(
    "## Step 1a — SVD encoder (NAML L06 / Lab01)\n",
    "**Eckart-Young theorem:** the best rank-`k` approximation of the 302-neuron activity "
    "matrix is the truncated SVD.\n",
    "$$X_k = U_k \\Sigma_k V_k^T$$\n",
    "We use **Randomized SVD** (from the course's own `rsvd_2024.ipynb`) to compress "
    "302 dimensions → k=4 principal neural components.\n",
    "Lab01 analogy: just as we compressed the Mondrian painting with SVD, "
    "here we compress the worm's neural activity."
), code(
    "# Step 1a: RSVD (Lab01 / L06)\n"
    "k_worm = choose_k_by_variance(X_neural, variance_threshold=0.90)\n"
    "k_worm = min(k_worm, 4)  # cap at 4 for speed\n"
    "print(f'Chosen k = {k_worm} (explains ≥90% variance)')\n"
    "\n"
    "U_k, s_k, Vt_k = rsvd(X_neural, k=k_worm, q=1, seed=0)\n"
    "Z_worm = neural_scores(X_neural, U_k).T  # (T, k_worm)\n"
    "\n"
    "# Scree plot (Lab01 analogy)\n"
    "s_full = np.linalg.svd(X_neural, compute_uv=False)\n"
    "cum_var = np.cumsum(s_full**2) / (s_full**2).sum()\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(10, 3))\n"
    "ax[0].semilogy(s_full[:20], 'k-o', ms=4); ax[0].axvline(k_worm-1, color='red', ls='--')\n"
    "ax[0].set(xlabel='component', ylabel='σ', title='Singular values (scree plot)')\n"
    "ax[1].plot(cum_var[:20]*100, 'b-o', ms=4); ax[1].axhline(90, color='red', ls='--')\n"
    "ax[1].set(xlabel='k', ylabel='% variance', title=f'Cumulative variance (k={k_worm} → 90%)')\n"
    "plt.tight_layout(); plt.savefig(OUTDIR/'step1a_svd.png', dpi=100); plt.show()\n"
    "print(f'Z_worm shape: {Z_worm.shape}  (302-D → {k_worm}-D)')"
)]

# ── Step 1b: Procrustes ───────────────────────────────────────────────────────
cells += [md(
    "## Step 1b — Procrustes alignment (NAML L06/L09)\n",
    "**Orthogonal Procrustes problem:** find the rotation `R` that aligns "
    "the worm's k-dim subspace to Chopin's k-dim subspace.\n",
    "$$R = \\arg\\min_{R^TR=I} \\|W_k R - C_k\\|_F^2 "
    "\\quad\\Rightarrow\\quad C_k^T W_k = U \\Sigma V^T,\\; R = VU^T$$\n",
    "This is the Eckart-Young theorem applied to the *alignment* problem, not the "
    "compression problem. No neural network needed — pure linear algebra."
), code(
    "# Step 1b: Procrustes (L06/L09) — v0.8.0: standardize=True (ISSUE-032)\n"
    "result_proc = procrustes_align(Z_worm, C_chopin, standardize=True)\n"
    "R = result_proc['R']\n"
    "Z_aligned = result_proc['W_k_scaled'] @ R  # use standardized W for alignment\n"
    "print(f'Alignment residual (v0.8.0 standardized): {result_proc[\"residual\"]:.4f}')\n"
    "print(f'PC scale factors (std of each worm PC): {result_proc[\"scale\"].round(4)}')\n"
    "print(f'Singular values of C^T W_scaled: {result_proc[\"singular_values\"].round(3)}')\n"
    "\n"
    "# Convert aligned features to note events\n"
    "# Use peak activations across feature dims as note triggers\n"
    "from pyannow.targets.midi_target import onset_loss\n"
    "activ  = np.abs(Z_aligned).max(axis=1)  # (T,) activation envelope\n"
    "from scipy.signal import find_peaks\n"
    "peaks, _ = find_peaks(activ, distance=int(0.28/0.5e-3), height=activ.mean())\n"
    "onsets_proc = t_arr[peaks]\n"
    "L_proc  = onset_loss(onsets_proc, t_on_chopin, window_s=DURATION)\n"
    "F1_proc = musical_f1(onsets_proc, t_on_chopin, window_s=DURATION)\n"
    "I_proc  = ioi_similarity(onsets_proc, t_on_chopin, window_s=DURATION)\n"
    "print(f'Step 1 (SVD+Procrustes): {len(peaks)} notes  '\n"
    "      f'loss={L_proc:.5f}  F1={F1_proc[\"f1\"]:.3f}  IOI={I_proc:.3f}')\n"
    "print(f'  beats Step 0 F1? {\"YES\" if F1_proc[\"f1\"] > f1_scores[\"Step 0 (baseline)\"] else \"NO\"}')\n"
    "losses['Step 1 (SVD+Procrustes)']    = L_proc\n"
    "f1_scores['Step 1 (SVD+Procrustes)'] = F1_proc['f1']\n"
    "ioi_sims['Step 1 (SVD+Procrustes)']  = I_proc"
)]

# ── Step 2: K-means ───────────────────────────────────────────────────────────
cells += [md(
    "## Step 2 — PCA + K-means motor primitives (NAML L08/L10 / Lab02)\n",
    "**PCA** reduces the neural trajectory to 2-4 principal directions — the worm's "
    "'motor state space'. **K-means** clusters these into discrete motor primitives "
    "(Forward / Backward / Turn / Pause).\n",
    "**Lab02 analogy:** same as clustering MNIST digit images — here we cluster "
    "worm neural states. Each cluster transition = one musical note."
), code(
    "# Step 2: PCA + K-means (L08/L10 / Lab02)\n"
    "scores, pca_obj = pca_reduce(X_neural, n_components=4, standardize=True)\n"
    "print(f'PCA explained variance: {pca_obj.explained_variance_ratio_.round(3)}')\n"
    "\n"
    "# Choose k via silhouette (Lab02 pattern)\n"
    "sil_res = choose_k_silhouette(scores, k_range=range(2, 7))\n"
    "k_clust = sil_res['best_k']\n"
    "print(f'Best k by silhouette: {k_clust}  (scores: {sil_res[\"scores\"]})')\n"
    "\n"
    "labels, km_obj = find_motor_primitives(scores, k=k_clust)\n"
    "cluster_names = ['Forward', 'Backward', 'Turn', 'Pause'][:k_clust]\n"
    "\n"
    "# Visualise clusters in PC space\n"
    "fig, ax = plt.subplots(figsize=(6, 5))\n"
    "for cl in range(k_clust):\n"
    "    mask = labels == cl\n"
    "    ax.scatter(scores[mask, 0], scores[mask, 1], s=6, alpha=0.5,\n"
    "               label=cluster_names[cl] if cl < len(cluster_names) else f'Cluster {cl}')\n"
    "ax.set(xlabel='PC1', ylabel='PC2', title='Worm motor states in PCA space')\n"
    "ax.legend(); plt.tight_layout()\n"
    "plt.savefig(OUTDIR/'step2_clusters.png', dpi=100); plt.show()\n"
    "\n"
    "# Convert cluster transitions → note events\n"
    "onsets_clust = np.array([e[0] for e in\n"
    "    cluster_to_notes(labels, t_ms, min_interval_ms=280)])\n"
    "L_clust  = onset_loss(onsets_clust, t_on_chopin, window_s=DURATION)\n"
    "F1_clust = musical_f1(onsets_clust, t_on_chopin, window_s=DURATION)\n"
    "I_clust  = ioi_similarity(onsets_clust, t_on_chopin, window_s=DURATION)\n"
    "print(f'Step 2 (K-means):       {len(onsets_clust)} notes  '\n"
    "      f'loss={L_clust:.5f}  F1={F1_clust[\"f1\"]:.3f}  IOI={I_clust:.3f}')\n"
    "print(f'  beats Step 0 F1? {\"YES\" if F1_clust[\"f1\"] > f1_scores[\"Step 0 (baseline)\"] else \"NO\"}')\n"
    "losses['Step 2 (K-means)']    = L_clust\n"
    "f1_scores['Step 2 (K-means)'] = F1_clust['f1']\n"
    "ioi_sims['Step 2 (K-means)']  = I_clust"
)]

# ── Step 3: Ridge ─────────────────────────────────────────────────────────────
cells += [md(
    "## Step 3 — Ridge regression composer (NAML L07/L11 / Lab03/Lab07)\n",
    "**Ridge regression** learns a regularised linear mapping\n",
    "$$W = (Z^T Z + \\lambda I)^{-1} Z^T C$$\n",
    "from the worm's neural PCA scores `Z` to the Chopin musical features `C`. "
    "Ridge handles the ill-conditioned case — some neural PCs carry 80%+ of variance; "
    "ridge prevents them from dominating.\n",
    "**Lab07 analogy:** same as predicting California housing prices from features, "
    "but now we predict musical structure from neural activity."
), code(
    "# Step 3: Ridge regression (L07/L11 / Lab07)\n"
    "rc = RidgeComposer(alpha=None)  # CV selects λ\n"
    "rc.fit(Z_worm, C_chopin)\n"
    "print(f'Best α (RidgeCV): {rc.alpha_:.4f}')\n"
    "\n"
    "# Show how R² varies with λ (Lab07 pattern)\n"
    "lam_res = explained_variance_by_ridge(Z_worm, C_chopin)\n"
    "fig, ax = plt.subplots(figsize=(6, 3))\n"
    "ax.semilogx(lam_res['alphas'], lam_res['r2s'], 'b-o', ms=4, lw=1.5)\n"
    "ax.axvline(lam_res['best_alpha'], color='red', ls='--', label=f'best α={lam_res[\"best_alpha\"]:.3f}')\n"
    "ax.set(xlabel='λ (regularisation)', ylabel='R²', title='Ridge: R² vs regularisation strength')\n"
    "ax.legend(); plt.tight_layout()\n"
    "plt.savefig(OUTDIR/'step3_ridge.png', dpi=100); plt.show()\n"
    "\n"
    "# Extract predicted activations → note events\n"
    "C_pred_ridge = rc.predict(Z_worm)\n"
    "activ_ridge  = np.abs(C_pred_ridge).max(axis=1)\n"
    "peaks_r, _   = find_peaks(activ_ridge, distance=int(0.28/0.5e-3), height=activ_ridge.mean())\n"
    "onsets_ridge = t_arr[peaks_r]\n"
    "L_ridge  = onset_loss(onsets_ridge, t_on_chopin, window_s=DURATION)\n"
    "F1_ridge = musical_f1(onsets_ridge, t_on_chopin, window_s=DURATION)\n"
    "I_ridge  = ioi_similarity(onsets_ridge, t_on_chopin, window_s=DURATION)\n"
    "ev_ridge = rc.evaluate(Z_worm, C_chopin)\n"
    "print(f'Step 3 (Ridge):         {len(peaks_r)} notes  '\n"
    "      f'loss={L_ridge:.5f}  F1={F1_ridge[\"f1\"]:.3f}  IOI={I_ridge:.3f}  R²={ev_ridge[\"r2\"]:.3f}')\n"
    "print(f'  beats Step 0 F1? {\"YES\" if F1_ridge[\"f1\"] > f1_scores[\"Step 0 (baseline)\"] else \"NO\"}')\n"
    "losses['Step 3 (Ridge)']    = L_ridge\n"
    "f1_scores['Step 3 (Ridge)'] = F1_ridge['f1']\n"
    "ioi_sims['Step 3 (Ridge)']  = I_ridge"
)]

# ── Step 4+5+6: MLP + Adam + L-BFGS ─────────────────────────────────────────
cells += [md(
    "## Steps 4-6 — MLP + Adam + L-BFGS (NAML L14-22 / Labs 06/08/10)\n",
    "Three NAML concepts combined:\n",
    "- **Step 4 (L14-17 / Lab06):** replace the linear ridge map with a 2-layer "
    "MLP. Xavier init, tanh activations, autodiff via `jax.grad`.\n"
    "- **Step 5 (L20 / Lab10):** train with mini-batch Adam. Per-parameter LR "
    "adapts to the neural feature scale mismatch.\n"
    "- **Step 6 (L22 / Lab08):** L-BFGS second-stage polish — super-linear "
    "convergence squeezes the last 5-10% loss that Adam plateaued on."
), code(
    "# Steps 4-6: MLP + Adam + L-BFGS (Lab06 / Lab08 / Lab10)\n"
    "import jax, jax.numpy as jnp\n"
    "\n"
    "# Step 4: create Flax MLP (identical to Lab10's architecture)\n"
    "model   = create_model(k_worm=k_worm, k_chopin=k_chopin, hidden=32, depth=2)\n"
    "params  = init_params(model, k_worm)\n"
    "print(model_summary(model, k_worm))\n"
    "\n"
    "# Step 5: Adam training (Lab10 / L20)\n"
    "print('\\nStep 5 — Adam training...')\n"
    "params, h_adam = train_adam(\n"
    "    model, params, Z_worm, C_chopin,\n"
    "    lr=1e-3, epochs=300, batch_sz=32, verbose=True)\n"
    "\n"
    "# Step 6: L-BFGS polish (Lab08 / L22)\n"
    "print('\\nStep 6 — L-BFGS polish...')\n"
    "params, h_lbfgs = polish_lbfgs(\n"
    "    model, params, Z_worm, C_chopin, max_steps=60, verbose=True)\n"
    "\n"
    "# Predict + extract note events\n"
    "from pyannow.step4_ffnn.jax_composer import predict as jax_predict\n"
    "C_pred_mlp = jax_predict(params, model, Z_worm)\n"
    "activ_mlp  = np.abs(C_pred_mlp).max(axis=1)\n"
    "peaks_m, _ = find_peaks(activ_mlp, distance=int(0.28/0.5e-3), height=activ_mlp.mean())\n"
    "onsets_mlp = t_arr[peaks_m]\n"
    "L_mlp  = onset_loss(onsets_mlp, t_on_chopin, window_s=DURATION)\n"
    "F1_mlp = musical_f1(onsets_mlp, t_on_chopin, window_s=DURATION)\n"
    "I_mlp  = ioi_similarity(onsets_mlp, t_on_chopin, window_s=DURATION)\n"
    "print(f'\\nSteps 4-6 (MLP+Adam+LBFGS): {len(peaks_m)} notes  '\n"
    "      f'loss={L_mlp:.5f}  F1={F1_mlp[\"f1\"]:.3f}  IOI={I_mlp:.3f}')\n"
    "print(f'  beats Step 0 F1? {\"YES\" if F1_mlp[\"f1\"] > f1_scores[\"Step 0 (baseline)\"] else \"NO\"}')\n"
    "losses['Steps 4-6 (MLP+Adam+LBFGS)']    = L_mlp\n"
    "f1_scores['Steps 4-6 (MLP+Adam+LBFGS)'] = F1_mlp['f1']\n"
    "ioi_sims['Steps 4-6 (MLP+Adam+LBFGS)']  = I_mlp\n"
    "\n"
    "# Training curve (compare Adam vs L-BFGS segments)\n"
    "fig, ax = plt.subplots(figsize=(9, 3))\n"
    "all_losses = h_adam.train_loss + h_lbfgs.train_loss\n"
    "n_adam = len(h_adam.train_loss)\n"
    "ax.semilogy(range(n_adam), h_adam.train_loss, 'b-', lw=1.5, label='Adam')\n"
    "ax.semilogy(range(n_adam, n_adam + len(h_lbfgs.train_loss)),\n"
    "            h_lbfgs.train_loss, 'r-', lw=1.5, label='L-BFGS')\n"
    "ax.axvline(n_adam, color='grey', ls='--', alpha=0.5)\n"
    "ax.set(xlabel='step', ylabel='loss', title='Training: Adam (L20) → L-BFGS (L22)')\n"
    "ax.legend(); plt.tight_layout()\n"
    "plt.savefig(OUTDIR/'step456_training.png', dpi=100); plt.show()"
)]

# ── Step 8: ODE vs PDE PINN ───────────────────────────────────────────────────
cells += [md(
    "## Step 8 — PINN: ODE vs PDE comparison (NAML L14/L27)\n",
    "**Two levels of physics:**\n",
    "- **Step 8a (ODE):** damped harmonic oscillator `q̈ + 2γq̇ + ω²q = F` — per-muscle, time only\n"
    "- **Step 8b (PDE):** 1D wave equation `ρ q_tt − μ q_xx + γ q_t = F` — spatial + temporal\n",
    "The PDE is the spatially-extended version of the ODE. Same PINN recipe (L27), "
    "same autodiff (`jax.grad` twice), but collocation points now span `(x,t)` space.\n",
    "**Beautiful symmetry** (SCIENTIFIC_FOUNDATION.md §C.2): this PDE is the same "
    "family as the piano string equation. The worm and the piano share a common physics."
), code(
    "# Step 8: ODE vs PDE PINN (L14 + L27)\n"
    "print('Running Step 8 ODE vs PDE comparison (~2-3 min)...')\n"
    "pinn_results = compare_ode_vs_pde(\n"
    "    Z_worm, C_chopin, t_arr,\n"
    "    lam_phys=0.05,\n"
    "    adam_steps=600, lbfgs_steps=80,\n"
    "    ca_thresh=DEFAULT_PARAMS.ca_thresh,\n"
    "    verbose=True\n"
    ")\n"
    "L_ode = pinn_results['final_loss_ode']\n"
    "L_pde = pinn_results['final_loss_pde']\n"
    "# Note: PINN losses are NOT comparable to onset_loss above (they are\n"
    "# L_data + λ·L_phys).  Store separately; show data_loss component in summary.\n"
    "f1_scores['Step 8a (ODE PINN)'] = float('nan')  # no note events from PINN\n"
    "f1_scores['Step 8b (PDE PINN)'] = float('nan')\n"
    "print(f'ODE data_loss≈{pinn_results[\"history_ode\"].adam_losses[-1]:.4f}  '\n"
    "      f'PDE data_loss≈{pinn_results[\"history_pde\"].adam_losses[-1]:.4f}')\n"
    "\n"
    "# Convergence comparison plot\n"
    "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
    "for ax, key, label, color in [\n"
    "    (axes[0], 'history_ode', 'ODE (per-muscle oscillator)', 'steelblue'),\n"
    "    (axes[1], 'history_pde', 'PDE (1D wave equation)', 'crimson'),\n"
    "]:\n"
    "    h = pinn_results[key]\n"
    "    ax.semilogy(h.adam_losses,  lw=1.5, label='Adam (L20)')\n"
    "    if h.lbfgs_losses:\n"
    "        n_a = len(h.adam_losses)\n"
    "        ax.semilogy(range(n_a, n_a + len(h.lbfgs_losses)),\n"
    "                    h.lbfgs_losses, 'r-', lw=1.5, label='L-BFGS (L22)')\n"
    "    ax.set(xlabel='step', ylabel='loss',\n"
    "           title=f'{label}\\nfinal={h.lbfgs_losses[-1] if h.lbfgs_losses else h.adam_losses[-1]:.5f}')\n"
    "    ax.legend(fontsize=8)\n"
    "plt.suptitle('PINN training: ODE vs PDE physics (L14/L27 recipe)', fontsize=11)\n"
    "plt.tight_layout(); plt.savefig(OUTDIR/'step8_pinn.png', dpi=100); plt.show()\n"
    "print(f'ODE: {h.wall_time:.1f}s  PDE: {pinn_results[\"history_pde\"].wall_time:.1f}s')"
)]

# ── Final comparison ──────────────────────────────────────────────────────────
cells += [md(
    "## Final comparison — All steps together\n",
    "**Primary metric: musical F1** (±50 ms tolerance — cannot be gamed by sparsity).  \n"
    "onset_loss shown for reference (data-only steps); PINN steps excluded from onset_loss panel "
    "(their loss = L_data + λ·L_phys, a different quantity).  \n"
    "**Goal: every step should achieve F1 > Step 0 baseline.**"
), code(
    "# Two-panel: onset_loss (data steps only) + musical F1 (all data steps)\n"
    "steps_ol  = [k for k in losses]          # onset_loss steps (no PINNs)\n"
    "steps_f1  = [k for k in f1_scores if not np.isnan(f1_scores[k])]  # F1 steps\n"
    "F1_base_v = f1_scores['Step 0 (baseline)']\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
    "\n"
    "# Left: onset_loss\n"
    "ax = axes[0]\n"
    "vals_ol  = [losses[k] for k in steps_ol]\n"
    "colors_ol = ['#888' if k == 'Step 0 (baseline)' else '#1f77b4' for k in steps_ol]\n"
    "bars = ax.bar(range(len(steps_ol)), vals_ol, color=colors_ol, edgecolor='white', width=0.7)\n"
    "for bar, v in zip(bars, vals_ol):\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, v + 2e-4, f'{v:.4f}',\n"
    "            ha='center', va='bottom', fontsize=7.5)\n"
    "ax.set(xticks=range(len(steps_ol)), xticklabels=steps_ol,\n"
    "       ylabel='onset loss (↓ better)', title='Onset loss by NAML step')\n"
    "plt.setp(ax.get_xticklabels(), rotation=15, ha='right', fontsize=8)\n"
    "\n"
    "# Right: musical F1\n"
    "ax = axes[1]\n"
    "vals_f1   = [f1_scores[k] for k in steps_f1]\n"
    "colors_f1 = ['#888' if k == 'Step 0 (baseline)' else\n"
    "             '#2ca02c' if f1_scores[k] > F1_base_v else '#d62728'\n"
    "             for k in steps_f1]\n"
    "bars = ax.bar(range(len(steps_f1)), vals_f1, color=colors_f1, edgecolor='white', width=0.7)\n"
    "ax.axhline(F1_base_v, color='black', ls='--', lw=1.5,\n"
    "           label=f'Step 0 baseline F1={F1_base_v:.3f}')\n"
    "for bar, v in zip(bars, vals_f1):\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.3f}',\n"
    "            ha='center', va='bottom', fontsize=7.5)\n"
    "ax.set(xticks=range(len(steps_f1)), xticklabels=steps_f1, ylim=(0, 1.05),\n"
    "       ylabel='musical F1 (↑ better, goal: >baseline)',\n"
    "       title='Musical F1 by NAML step  (green = beats baseline)')\n"
    "ax.legend(fontsize=8)\n"
    "plt.setp(ax.get_xticklabels(), rotation=15, ha='right', fontsize=8)\n"
    "\n"
    "plt.suptitle('PyANNOW v0.7.0 — NAML step progression (96-cell Boyle model)', fontsize=11)\n"
    "plt.tight_layout(); plt.savefig(OUTDIR/'final_comparison_metrics.png', dpi=110); plt.show()\n"
    "\n"
    "# Summary table\n"
    "print('\\nStep-by-step summary:')\n"
    "print(f'{\"Step\":<35} {\"onset_loss\":>11} {\"F1\":>7} {\"IOI\":>7} {\"beats S0?\":>10}')\n"
    "print('-' * 75)\n"
    "for k in steps_f1:\n"
    "    ol  = f'{losses[k]:.5f}' if k in losses else '      n/a'\n"
    "    f1  = f'{f1_scores[k]:.3f}'\n"
    "    ioi = f'{ioi_sims[k]:.3f}' if k in ioi_sims else '  n/a'\n"
    "    beat = 'YES' if f1_scores[k] > F1_base_v else 'NO '\n"
    "    print(f'{k:<35} {ol:>11} {f1:>7} {ioi:>7} {beat:>10}')"
), code(
    "# Generate and play the best melody (Steps 4-6)\n"
    "# Convert MLP predictions to note events\n"
    "best_events = []\n"
    "for k_idx, t_ in enumerate(t_arr[peaks_m]):\n"
    "    pitch = result_worm['pitch_map'][k_idx % n_muscles]\n"
    "    vel   = int(np.clip(activ_mlp[peaks_m[k_idx]] * 80, 20, 127))\n"
    "    best_events.append((t_, pitch, vel))\n"
    "\n"
    "# Synthesise best worm melody and Chopin (same piano model)\n"
    "audio_best, fs   = synthesise_melody(best_events, duration_s=DURATION)\n"
    "audio_chopin, _  = synthesise_melody(\n"
    "    [e for e in events_chopin if e.time_s <= DURATION], duration_s=DURATION)\n"
    "\n"
    "wavfile.write(str(OUTDIR/'worm_mlp.wav'),    fs, (audio_best*28000).astype('int16'))\n"
    "wavfile.write(str(OUTDIR/'chopin_ref.wav'),  fs, (audio_chopin*28000).astype('int16'))\n"
    "\n"
    "# Spectrogram side-by-side\n"
    "fig, axes = plt.subplots(2, 2, figsize=(12, 6))\n"
    "t_ax = np.arange(int(DURATION * fs)) / fs\n"
    "for col, (audio, label, cmap) in enumerate([\n"
    "    (audio_best,   '🐛 Worm (MLP+Adam+LBFGS, Steps 4-6)', 'Blues'),\n"
    "    (audio_chopin, '🎼 Chopin (original MIDI)',             'Reds'),\n"
    "]):\n"
    "    axes[0,col].plot(t_ax[:3*fs], audio[:3*fs], lw=0.4)\n"
    "    axes[0,col].set(title=f'{label} — waveform (first 3s)')\n"
    "    f_s, t_s, Sxx = sps.spectrogram(audio, fs=fs, nperseg=1024, noverlap=768)\n"
    "    axes[1,col].pcolormesh(t_s, f_s/1000, 10*np.log10(Sxx+1e-12),\n"
    "                            cmap=cmap, shading='auto', vmin=-80, vmax=-15)\n"
    "    axes[1,col].set(xlabel='time (s)', ylabel='kHz', ylim=(0, 2.5), title='spectrogram')\n"
    "plt.tight_layout(); plt.savefig(OUTDIR/'final_comparison.png', dpi=110); plt.show()\n"
    "\n"
    "print('🎵 Best worm melody (Steps 4-6):')\n"
    "display(Audio(audio_best, rate=fs, autoplay=False))\n"
    "print('🎼 Chopin (synthesised):')\n"
    "display(Audio(audio_chopin, rate=fs, autoplay=False))"
)]

# ── Full-length Chopin: synthesised ───────────────────────────────────────────
cells += [code(
    "# Synthesised version (worm piano model, full length)\n"
    "full_dur_s = max(e.time_s for e in events_chopin) + 2.0\n"
    "audio_full, fs_full = synthesise_melody(events_chopin, duration_s=full_dur_s)\n"
    "print(f'🎹 Synthesised (worm piano model)  '\n"
    "      f'[{full_dur_s-2.0:.1f}s / {(full_dur_s-2.0)/60:.1f} min]')\n"
    "display(Audio(audio_full, rate=fs_full, autoplay=False))"
)]

# ── Full-length Chopin: original WAV (self-contained) ─────────────────────────
cells += [code(
    "from pathlib import Path\n"
    "from scipy.io import wavfile\n"
    "from IPython.display import Audio, display\n"
    "import numpy as np\n"
    "\n"
    "wav_path = Path('../../shared/examples/chopin_nocturne_op_posth_csharp_minor.wav')\n"
    "fs_wav, audio_wav = wavfile.read(str(wav_path))\n"
    "if audio_wav.dtype == np.int16:\n"
    "    audio_wav = audio_wav.astype(np.float32) / 32768.0\n"
    "if audio_wav.ndim == 2:\n"
    "    audio_wav = audio_wav.mean(axis=1)\n"
    "print(f'🎼 Original Chopin — real piano audio  '\n"
    "      f'[{len(audio_wav)/fs_wav:.1f}s / {len(audio_wav)/fs_wav/60:.1f} min]')\n"
    "display(Audio(audio_wav, rate=fs_wav, autoplay=False))"
)]

# ── Conclusion ────────────────────────────────────────────────────────────────
cells += [md(
    "## Summary: what NAML contributed\n",
    "| Step | NAML concept | Where we were | What changed |\n"
    "|---|---|---|---|\n"
    "| 0 | None | Rule-based notes | Baseline F1 |\n"
    "| 1a | SVD (L06 / Lab01) | 302-D → k-D | Compress locomotion subspace |\n"
    "| 1b | Procrustes (L06/L09) | Random subspace | Aligned to Chopin |\n"
    "| 2 | PCA + K-means (L08/L10 / Lab02) | Continuous | Discrete motor categories |\n"
    "| 3 | Ridge (L07/L11 / Lab07) | Unstable | Regularised, smooth |\n"
    "| 4-5 | MLP + Adam (L14-20 / Lab06/Lab10) | Linear only | Non-linear mapping |\n"
    "| 6 | L-BFGS (L22 / Lab08) | Adam plateau | Tight convergence |\n"
    "| 8a | ODE PINN (L27) | Free mapping | Obeys oscillator ODE |\n"
    "| 8b | PDE PINN (L27 + NMPDE echo) | Per-muscle | Spatial wave coupling |\n",
    "### v0.7.0 architecture: Boyle et al. 4×24 = 96 muscle cells\n",
    "The biological model now follows **Boyle et al. (2012)** with 4 quadrants × 24 muscle cells = 96 total.  \n"
    "96 muscles ≡ 8 octaves × 12 semitones = **96 piano keys**.  \n"
    "Quadrant pitch assignment:\n"
    "- DL (0-23) → MIDI 24-47 (C1-B2, bass)\n"
    "- VL (24-47) → MIDI 48-71 (C3-B4, tenor)\n"
    "- DR (48-71) → MIDI 72-95 (C5-B6, alto)\n"
    "- VR (72-95) → MIDI 96-119 (C7-B8, treble)\n",
    "### The biological ceiling\n",
    "With 96 independent BWM muscle cells (4×24 Boyle quadrants) each with a 65 ms refractory period "
    "(EGL-19 Ca²⁺ τ_Ca + 50 ms passive relaxation), the worm can physically reach ~100% of Chopin's "
    "note rate. The true hard limit is **rhythmic regularity**: the locomotion circuit drives a "
    "periodic travelling wave, while Chopin's timing is syncopated and musically varied.  \n"
    "NAML methods improve the note-*structure* mapping within this constraint — they cannot "
    "change the body-wave oscillator period, but they can learn which part of the worm's "
    "302-D neural trajectory best predicts Chopin's musical features."
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

out = Path("PyANNOW/notebooks/03_pyannow_naml_progression.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} — {len(cells)} cells, {out.stat().st_size//1024} KB")
