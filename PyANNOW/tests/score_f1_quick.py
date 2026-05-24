"""Quick F1 scoring test for all PyANNOW pipeline steps.

Run with:
    /Users/vghayoomie/miniconda3/envs/naml-pinn-312/bin/python3 tests/score_f1_quick.py

Expected runtime: ~90-120s with Step 10 (adds full 229s worm simulation).
Set SKIP_STEP10=True to revert to ~60-90s (no PINN, MLP capped at 50 epochs).
"""
import warnings, sys, time
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
from pathlib import Path

# ── Imports ──────────────────────────────────────────────────────────────────
from pyannow.targets.midi_target import (
    parse_midi, note_onsets, onset_loss, musical_f1, ioi_similarity,
    calibrated_onset_detect,
)
from pyannow.composer.worm_optimizer_fast import (
    run_forward_fast, onsets_from_result, MUSCLE_PITCHES_96,
)
from pyannow.composer.worm_optimizer import generate_neural_activity_302
from pyannow.ion_channels.celegans_hh import DEFAULT_PARAMS
from pyannow.step1_svd.encoder import rsvd, neural_scores, choose_k_by_variance
from pyannow.step1_svd.procrustes import (
    procrustes_align, apply_rotation, build_chopin_features, chopin_cumvar,
)
from pyannow.step2_clustering.motor_primitives import (
    pca_reduce, find_motor_primitives, cluster_to_notes,
)
from pyannow.step3_regression.ridge_composer import RidgeComposer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

MIDI_PATH = Path('/Users/vghayoomie/git/wormuse/shared/examples/chopin_nocturne_op_posth_csharp_minor.mid')
DURATION  = 30.0

print("=" * 64)
print("PyANNOW v0.9.0 — Quick F1 benchmark")
print("=" * 64)
t_total = time.perf_counter()

# ── 1. Worm simulation ───────────────────────────────────────────────────────
print("\n[Setup] Running 96-cell worm simulation...")
t0 = time.perf_counter()
result_worm = run_forward_fast(
    DEFAULT_PARAMS, duration_s=DURATION, dt_ms=0.5,
    drive_freq_hz=1.5, drive_amplitude=12.0, random_seed=42)
V_mus       = result_worm['V_muscles']
t_ms        = result_worm['t_arr_ms']
t_arr       = t_ms * 1e-3
T_pts       = len(t_arr)
worm_events = result_worm['note_onsets_s']
print(f"  Worm: T={T_pts} pts, {len(worm_events)} events  [{time.perf_counter()-t0:.1f}s]")

# ── 2. 302-neuron activity ───────────────────────────────────────────────────
print("[Setup] Generating 302-neuron activity...")
t0 = time.perf_counter()
X_neural = generate_neural_activity_302(
    n_steps=T_pts, dt_ms=0.5, drive_freq_hz=1.5, seed=42)
print(f"  X_neural: {X_neural.shape}  [{time.perf_counter()-t0:.1f}s]")

# ── 3. Chopin MIDI ───────────────────────────────────────────────────────────
print("[Setup] Parsing Chopin MIDI...")
events_chopin, bpm = parse_midi(MIDI_PATH)
t_on_chopin = note_onsets(events_chopin, clip_s=DURATION)
C_chopin = build_chopin_features(
    events_chopin, duration_s=DURATION, n_bins=T_pts,
    k_chopin=None, var_threshold=0.90)
k_chopin = C_chopin.shape[1]
print(f"  Chopin: {len(t_on_chopin)} onsets, BPM={bpm:.0f}, "
      f"C_chopin={C_chopin.shape}")

# ── 4. Worm PCA (SVD) ────────────────────────────────────────────────────────
print("[Setup] Computing worm PCA...")
t0 = time.perf_counter()
k_worm = choose_k_by_variance(X_neural, variance_threshold=0.90)
U_k, s_k, Vt_k = rsvd(X_neural, k=k_worm, q=1, seed=0)
Z_worm = neural_scores(X_neural, U_k).T    # (T, k_worm)
print(f"  k_worm={k_worm}, Z_worm={Z_worm.shape}  [{time.perf_counter()-t0:.1f}s]")

# ── Metric helpers ───────────────────────────────────────────────────────────
losses, f1_scores, ioi_sims = {}, {}, {}

BASELINE_KEY = "Step 0 (body-wave)"

def score(label, onsets):
    L  = onset_loss(onsets, t_on_chopin, window_s=DURATION)
    F1 = musical_f1(onsets, t_on_chopin, window_s=DURATION)
    I  = ioi_similarity(onsets, t_on_chopin, window_s=DURATION)
    losses[label]    = L
    f1_scores[label] = F1['f1']
    ioi_sims[label]  = I
    baseline = f1_scores.get(BASELINE_KEY, None)
    if label == BASELINE_KEY:
        tag = " ← baseline"
    elif baseline is not None and F1['f1'] > baseline:
        tag = " ✓ beats baseline"
    elif baseline is not None:
        tag = " ✗ below baseline"
    else:
        tag = ""
    print(f"  {label:38s}  loss={L:.5f}  F1={F1['f1']:.6f}  IOI={I:.3f}{tag}")

def cal(activ, label):
    """calibrated_onset_detect wrapper."""
    return calibrated_onset_detect(
        activ, t_on_chopin, t_arr, tol_s=0.05, refractory_s=0.28)

print("\n" + "-" * 64)
print("STEP RESULTS")
print("-" * 64)

# ── Step 0: body-wave baseline ───────────────────────────────────────────────
onsets_0 = np.array([ev[0] for ev in worm_events if ev[0] <= DURATION])
score("Step 0 (body-wave)", onsets_0)

# ── Step 1a: SVD / PCA ───────────────────────────────────────────────────────
t0 = time.perf_counter()
a1 = np.abs(Z_worm[:, 0])   # first PC amplitude
onsets_1 = cal(a1, "Step 1a")
score("Step 1a (SVD/RSVD)", onsets_1)
print(f"    [{time.perf_counter()-t0:.1f}s]")

# ── Step 1b: Procrustes alignment ────────────────────────────────────────────
t0 = time.perf_counter()
result_proc  = procrustes_align(Z_worm, C_chopin, standardize=True)
Z_aligned    = apply_rotation(result_proc['W_k_scaled'], result_proc['R'])  # (T, k_chopin)
activ_proc   = np.abs(Z_aligned).max(axis=1)
onsets_1b   = cal(activ_proc, "Step 1b")
score("Step 1b (Procrustes)", onsets_1b)
print(f"    [{time.perf_counter()-t0:.1f}s]")

# ── Step 2: Motor primitives clustering ──────────────────────────────────────
t0 = time.perf_counter()
scores_pca, pca_obj = pca_reduce(X_neural, n_components=4, standardize=True)
labels_2, km_obj = find_motor_primitives(scores_pca, k=4)
# Reconstruct activity = distance from cluster centroid (anomaly signal)
centers = km_obj.cluster_centers_[labels_2]
activ2  = np.linalg.norm(scores_pca - centers, axis=1)
onsets_2 = cal(activ2, "Step 2")
score("Step 2 (Motor primitives)", onsets_2)
print(f"    [{time.perf_counter()-t0:.1f}s]")

# ── Step 3: Ridge regression ──────────────────────────────────────────────────
t0 = time.perf_counter()
rc = RidgeComposer(alpha=None)
rc.fit(Z_worm, C_chopin)
C_pred_ridge  = rc.predict(Z_worm)
activ_ridge   = np.abs(C_pred_ridge).max(axis=1)
onsets_3      = cal(activ_ridge, "Step 3")
score("Step 3 (Ridge)", onsets_3)
print(f"    best_α={rc.alpha_:.4f}  R²={rc.evaluate(Z_worm, C_chopin)['r2']:.3f}  [{time.perf_counter()-t0:.1f}s]")

# ── Step 7: RandomForest (supervised, NEW in v0.9.0) ─────────────────────────
t0 = time.perf_counter()
tol_steps = int(0.05 / (t_arr[1] - t_arr[0]))
y_onset   = np.zeros(T_pts, dtype=int)
for t_on in t_on_chopin:
    idx = int(np.searchsorted(t_arr, t_on))
    lo  = max(0, idx - tol_steps)
    hi  = min(T_pts, idx + tol_steps + 1)
    y_onset[lo:hi] = 1

Zs = StandardScaler().fit_transform(Z_worm)
rf = RandomForestClassifier(
    n_estimators=100, class_weight='balanced',
    oob_score=True, n_jobs=2, random_state=0)
rf.fit(Zs, y_onset)
probs_rf  = rf.predict_proba(Zs)[:, 1]
onsets_7  = cal(probs_rf, "Step 7")
score("Step 7 (RandomForest)", onsets_7)
print(f"    OOB={rf.oob_score_:.3f}  [{time.perf_counter()-t0:.1f}s]")

# ── Step 4: JAX MLP (lightweight, 50 epochs) ─────────────────────────────────
# Step 4 (JAX MLP): ~175s due to JAX XLA compile overhead even at 50 epochs.
# Run only when explicitly enabled (SKIP_JAX_MLP=False) — notebook has SKIP_PINN flag.
SKIP_JAX_MLP = True
if not SKIP_JAX_MLP:
    try:
        from pyannow.step4_ffnn.jax_composer import create_model, init_params, predict
        from pyannow.step5_training.adam_trainer import train_adam
        t0 = time.perf_counter()
        model  = create_model(k_worm=k_worm, k_chopin=k_chopin, hidden=64, depth=2)
        params = init_params(model, k_worm=k_worm, seed=0)
        params, _ = train_adam(
            model, params, Z_worm, C_chopin, epochs=50, verbose=False)
        C_pred_mlp = predict(params, model, Z_worm)
        activ_mlp  = np.abs(C_pred_mlp).max(axis=1)
        onsets_4   = cal(activ_mlp, "Step 4")
        score("Step 4 (MLP Adam 50ep)", onsets_4)
        print(f"    [{time.perf_counter()-t0:.1f}s]")
    except Exception as e:
        print(f"  Step 4 SKIPPED: {e}")
else:
    print("  Step 4 (JAX MLP): skipped — set SKIP_JAX_MLP=False to run (~3 min)")

# ── Step 4 MLP timing note ────────────────────────────────────────────────────
# JAX first-call compile overhead dominates at low epoch count.
# Workaround: use sklearn MLPClassifier for fast iteration (Step 9),
# then optionally warm-start JAX once compiled.

# ── Step 5: L-BFGS polish (skip if slow) ─────────────────────────────────────
# Skipped in quick test — takes 2-3 min

# ── Step 6: L-BFGS only (no MLP warmup) ─────────────────────────────────────
# Skipped in quick test

# ── Step 9: Worm + Fourier time hybrid (ISSUE-042, inspired by NB04) ─────────
# Idea: concatenate Fourier time embeddings (from NB04) with worm PCA scores,
# then train a fast sklearn MLP onset-classifier. This blends temporal structure
# (NB04 approach) with the biological worm signal, targeting F1 > RF & NB04.
try:
    from sklearn.neural_network import MLPClassifier
    t0 = time.perf_counter()
    print("\n[Step 9] Worm+Time hybrid MLP (sklearn, fast)...")

    # Fourier time embeddings at worm resolution (0.5ms)
    phase = t_arr / t_arr[-1]                           # (T,) in [0,1]
    n_harmonics = 12
    time_feats = [phase[:, None]]
    for k_h in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * k_h * phase
        time_feats.append(np.sin(angle)[:, None])
        time_feats.append(np.cos(angle)[:, None])
    # Beat-phase (BPM from MIDI)
    beat_phase = t_arr * bpm / 60.0
    time_feats.append(np.sin(2 * np.pi * beat_phase)[:, None])
    time_feats.append(np.cos(2 * np.pi * beat_phase)[:, None])
    T_feats = np.concatenate(time_feats, axis=1)       # (T, 27)

    # Concatenate with standardized worm PCA scores
    scaler9 = StandardScaler()                          # kept for Step 10 reuse
    Zs9 = scaler9.fit_transform(Z_worm)                # (T, k_worm)
    X9  = np.concatenate([T_feats, Zs9], axis=1)      # (T, 27+k_worm)

    # Subsample to 10000 pts for MLP speed (uniform)
    sub_n = min(10000, T_pts)
    idx_s = np.linspace(0, T_pts - 1, sub_n, dtype=int)
    X9_s  = X9[idx_s]
    y9_s  = y_onset[idx_s]                            # reuse onset labels from Step 7

    # Balance classes via sample_weight (MLPClassifier has no class_weight param)
    pos_ratio = y9_s.sum() / max(1, len(y9_s) - y9_s.sum())
    sw9 = np.where(y9_s == 1, 1.0 / max(pos_ratio, 1e-6), 1.0)

    mlp9 = MLPClassifier(
        hidden_layer_sizes=(128, 64), activation='relu',
        solver='adam', max_iter=200, early_stopping=True,
        n_iter_no_change=15, random_state=0, verbose=False)
    mlp9.fit(X9_s, y9_s, sample_weight=sw9)

    # Predict full sequence
    probs9  = mlp9.predict_proba(X9)[:, 1]
    onsets9 = cal(probs9, "Step 9")
    score("Step 9 (Worm+Time hybrid MLP)", onsets9)
    print(f"    iters={mlp9.n_iter_}  [{time.perf_counter()-t0:.1f}s]")
except Exception as e:
    print(f"  Step 9 SKIPPED: {e}")

# ── Step 9b: Physics-residual enhanced MLP (ISSUE-042c, fast variant) ────────
# Extends Step 9 by appending the ODE residual r(t) = q̈ + 2γq̇ + ω²q as
# extra features.  At muscle contraction → relaxation transitions (onsets),
# the forcing term F changes sign, producing a spike in r(t) that the MLP
# can exploit — free physics information at near-zero cost.
try:
    from sklearn.neural_network import MLPClassifier as _MLPClassifier_9b
    t0 = time.perf_counter()
    print("\n[Step 9b] Physics-residual enhanced MLP (ODE features)...")

    dt_step = t_arr[1] - t_arr[0]   # 0.5 ms
    # Numerical derivatives of worm PCA scores
    dZw    = np.gradient(Z_worm, dt_step, axis=0)    # dq/dt  (T, k_worm)
    d2Zw   = np.gradient(dZw,   dt_step, axis=0)    # d²q/dt² (T, k_worm)
    # ODE residual: q̈ + 2·0.3·q̇ + 2.5²·q  (should be ~F_neural; spike at onset)
    ode_res   = d2Zw + 2.0 * 0.3 * dZw + 2.5**2 * Z_worm    # (T, k_worm)
    ode_res_s = StandardScaler().fit_transform(ode_res)

    # Extended features: Fourier time (27-D) + worm PCA (k-D) + ODE residual (k-D)
    X9b = np.hstack([T_feats, Zs9, ode_res_s])               # (T, 27+2·k)

    sub_n9b    = min(10000, T_pts)
    idx9b      = np.linspace(0, T_pts - 1, sub_n9b, dtype=int)
    X9b_s, y9b_s = X9b[idx9b], y_onset[idx9b]

    pos_ratio9b = y9b_s.sum() / max(1, len(y9b_s) - y9b_s.sum())
    sw9b = np.where(y9b_s == 1, 1.0 / max(pos_ratio9b, 1e-6), 1.0)

    mlp9b = _MLPClassifier_9b(
        hidden_layer_sizes=(128, 64), activation='relu',
        solver='adam', max_iter=200, early_stopping=True,
        n_iter_no_change=15, random_state=0, verbose=False)
    mlp9b.fit(X9b_s, y9b_s, sample_weight=sw9b)

    probs9b  = mlp9b.predict_proba(X9b)[:, 1]
    onsets9b = cal(probs9b, "Step 9b")
    score("Step 9b (Worm+Time+Physics MLP)", onsets9b)
    print(f"    iters={mlp9b.n_iter_}  [{time.perf_counter()-t0:.1f}s]")
except Exception as e:
    print(f"  Step 9b SKIPPED: {e}")

# ── Step 8c: PINN-Classifier (BCE + ODE physics) — guarded ───────────────────
# PINN-Classifier replaces the regression data loss (MSE vs Chopin features)
# with a classification loss (BCE vs y_onset), keeping the ODE physics residual.
# This is the principled union of Step 7/9 (classification) and Step 8 (physics).
# Runtime: ~5 min (JAX XLA compile + Adam 600 + L-BFGS 60).
SKIP_PINN8C = True
if not SKIP_PINN8C:
    try:
        from pyannow.step8_pinn.locomotion_pinn import run_pinn_classifier
        t0 = time.perf_counter()
        print("\n[Step 8c] PINN-Classifier (BCE + ODE, ~5 min)...")
        result_8c = run_pinn_classifier(
            Z_worm, y_onset.astype(float), t_arr,
            bpm=bpm, n_harmonics=12,
            max_train_pts=10000, lam_phys=0.02,
            gamma=0.3, omega=2.5,
            adam_steps=300, lbfgs_steps=50,
            seed=0, verbose=True)
        probs8c  = result_8c['probs']
        onsets8c = cal(probs8c, "Step 8c")
        score("Step 8c (PINN-Classifier)", onsets8c)
        print(f"    [{time.perf_counter()-t0:.1f}s]")
    except Exception as e:
        print(f"  Step 8c SKIPPED: {e}")
else:
    print("  Step 8c (PINN-Classifier): skipped — set SKIP_PINN8C=False (~5 min)")

# ── Step 10: Full-piece generalization (ISSUE-042d) ──────────────────────────
# Apply the saved mlp9 to the full 227s Chopin piece (walk-forward eval).
# mlp9 was trained on DURATION=30s; Step 10 tests generalisation to all 1000 notes.
# NOTE: runs a full 229s worm simulation — adds ~30s to the test.
SKIP_STEP10 = False
if not SKIP_STEP10 and 'mlp9' in dir():
    try:
        t0 = time.perf_counter()
        print("\n[Step 10] Full-piece generalization (227s Chopin)...")
        t_on_all = note_onsets(events_chopin, clip_s=10000)    # all 1000 notes
        T_full_s = float(t_on_all[-1] + 2.0)                  # ≈ 229s

        # Full worm simulation
        result_s10 = run_forward_fast(
            DEFAULT_PARAMS, duration_s=T_full_s, dt_ms=0.5,
            drive_freq_hz=1.5, drive_amplitude=12.0, random_seed=42)
        t_full     = result_s10['t_arr_ms'] * 1e-3
        T_full_pts = len(t_full)
        print(f"    Sim: T={T_full_pts} pts ({T_full_s:.0f}s)  [{time.perf_counter()-t0:.1f}s]")

        # 302-neuron activity + project onto same U_k
        X_neural_full = generate_neural_activity_302(
            n_steps=T_full_pts, dt_ms=0.5, drive_freq_hz=1.5, seed=42)
        Z_worm_full   = neural_scores(X_neural_full, U_k).T   # (T_full_pts, k_worm)
        Zs_full       = scaler9.transform(Z_worm_full)         # reuse train scaler

        # Fourier + beat features (T_full_s normalization for positional context)
        phase_full  = t_full / T_full_s
        tfeats_full = [phase_full[:, None]]
        for _kh in range(1, n_harmonics + 1):
            _ang = 2.0 * np.pi * _kh * phase_full
            tfeats_full += [np.sin(_ang)[:, None], np.cos(_ang)[:, None]]
        bp_full     = t_full * bpm / 60.0
        tfeats_full += [np.sin(2 * np.pi * bp_full)[:, None],
                        np.cos(2 * np.pi * bp_full)[:, None]]
        T_feats_full = np.concatenate(tfeats_full, axis=1)
        X10          = np.hstack([T_feats_full, Zs_full])

        # Apply saved mlp9 — no retraining
        probs10 = mlp9.predict_proba(X10)[:, 1]

        # Overall full-piece F1
        onsets10_full = calibrated_onset_detect(
            probs10, t_on_all, t_full, tol_s=0.05, refractory_s=0.28)
        L10_full  = onset_loss(onsets10_full, t_on_all, window_s=T_full_s)
        F1_10_full = musical_f1(onsets10_full, t_on_all, window_s=T_full_s)
        I10_full   = ioi_similarity(onsets10_full, t_on_all, window_s=T_full_s)

        # Holdout F1 (notes beyond training DURATION)
        t_on_hold  = t_on_all[t_on_all >= DURATION]
        _hm        = t_full >= DURATION
        _h_det     = calibrated_onset_detect(
            probs10[_hm], t_on_hold - DURATION, t_full[_hm] - DURATION,
            tol_s=0.05, refractory_s=0.28)
        F1_10_hold = musical_f1(_h_det, t_on_hold - DURATION,
                                window_s=T_full_s - DURATION)

        losses["Step 10 (Full-piece holdout)"]    = L10_full
        f1_scores["Step 10 (Full-piece holdout)"] = F1_10_hold['f1']
        ioi_sims["Step 10 (Full-piece holdout)"]  = I10_full
        gap = f1_scores.get("Step 9 (Worm+Time hybrid MLP)", 0) - F1_10_hold['f1']
        baseline = f1_scores.get(BASELINE_KEY, 0)
        tag = " ✓ beats baseline" if F1_10_hold['f1'] > baseline else " ✗ below baseline"
        print(f"  {'Step 10 (Full-piece holdout)':38s}  loss={L10_full:.5f}  "
              f"F1={F1_10_hold['f1']:.6f}  IOI={I10_full:.3f}{tag}")
        print(f"    train F1={f1_scores.get('Step 9 (Worm+Time hybrid MLP)',0):.6f}  "
              f"holdout F1={F1_10_hold['f1']:.6f}  gap={gap:+.6f}  "
              f"[{time.perf_counter()-t0:.1f}s]")
    except Exception as e:
        print(f"  Step 10 SKIPPED: {e}")
elif not SKIP_STEP10:
    print("  Step 10: skipped — mlp9 not available (Step 9 failed)")
else:
    print("  Step 10: skipped — SKIP_STEP10=True")

# ── Score network approach (NB04) on pure time features ─────────────────────
# This is a reference ceiling: what can a network achieve if it sees time directly?
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))  # PyANNOW root
    from chopin_score_net.data import build_score_dataset, best_frame_threshold, roll_to_note_events
    from chopin_score_net.train import train_score_model, predict_probabilities
    t0 = time.perf_counter()
    print("\n[NB04 ref] Training Fourier→score network (~60s)...")
    dataset = build_score_dataset(MIDI_PATH, resolution_s=0.02, n_harmonics=12)
    model_s, params_s, hist_s = train_score_model(
        dataset.features, dataset.target_roll,
        width=128, depth=2, lr=1e-3, epochs=80, batch_size=512, patience=15,
        verbose=False)
    probs_s = predict_probabilities(model_s, params_s, dataset.features)
    thr_s, frame_f1_s = best_frame_threshold(probs_s, dataset.target_roll)
    pred_events = roll_to_note_events(
        dataset.times, dataset.pitches, probs_s, threshold=thr_s, min_bins=2)
    onsets_s04 = np.array([ev.time_s for ev in pred_events if ev.time_s <= DURATION])
    score("NB04 (Fourier score-net)", onsets_s04)
    print(f"    frame_F1={frame_f1_s:.4f}  [{time.perf_counter()-t0:.1f}s]")
except Exception as e:
    print(f"  NB04 reference SKIPPED: {e}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("SUMMARY")
print("=" * 64)
baseline_f1 = f1_scores.get(BASELINE_KEY, 0)
print(f"{'Step':38s}  {'F1':>10s}  {'vs baseline':>12s}")
print("-" * 68)
for k, v in f1_scores.items():
    if k == BASELINE_KEY:
        tag = " ← baseline"
    elif v > baseline_f1:
        tag = f" +{v - baseline_f1:+.6f}"
    else:
        tag = f" {v - baseline_f1:+.6f}"
    print(f"  {k:38s}  {v:.6f}{tag}")
print(f"\nTotal elapsed: {time.perf_counter()-t_total:.1f}s")
