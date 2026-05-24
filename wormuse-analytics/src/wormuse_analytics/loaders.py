"""Loaders — re-run PyANNOW steps and cache their outputs.

Not tied to an AppStat lecture; this is infrastructure. Centralises every
import from `pyannow` so the rest of the package never touches PyANNOW
internals directly.

Cache layout (relative to wormuse-analytics/):
    data/cached_step_outputs.npz
        - t_arr               (T,)    time axis (s)
        - X_neural            (302,T) synthetic 302-neuron activity
        - C_chopin            (T,k_chopin) Chopin feature matrix
        - chopin_onsets       (M,)    Chopin onset times (s)
        - onsets_step0..      (Ni,)   onsets produced by each step
        - activ_step1..       (T,)    activation envelopes per step (for ROC/PR)
        - meta                dict    {DURATION, n_neurons, k_worm, k_chopin, ...}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings

import numpy as np


@dataclass
class StepOutputs:
    """Container for everything one step produces.

    Onsets are the *event* output (used by F1 / IOI metrics).
    Activations are the *continuous* output (used by ROC / PR curves).
    """
    name:        str        # e.g. "Step 0 (baseline)"
    onsets:      np.ndarray # onset times in seconds, (Ni,)
    activation:  np.ndarray | None  # smooth activation per timestep, (T,), or None for Step 0
    n_notes:     int


def _default_cache_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "cached_step_outputs.npz"


def run_pyannow_pipeline(
    duration_s: float = 10.0,
    midi_rel: str = "../../shared/examples/chopin_nocturne_op_posth_csharp_minor.mid",
    seed: int = 0,
    cache_path: Path | None = None,
    force: bool = False,
) -> dict:
    """Run PyANNOW steps 0-6 (skipping the expensive PINN step 8) and cache outputs.

    Re-runs PyANNOW deterministically with `random_seed=seed`. Caches to
    `cache_path` so subsequent calls are instant. Set `force=True` to overwrite.

    Returns a dict with keys: t_arr, X_neural, C_chopin, chopin_onsets, steps,
    where `steps` is a list of StepOutputs.
    """
    cache_path = cache_path or _default_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force:
        return _load_cache(cache_path)

    # Late imports so module imports without pyannow installed (e.g. for doc render)
    from pyannow.targets.midi_target import parse_midi, note_onsets
    from pyannow.composer.worm_optimizer_fast import (
        run_forward_fast, onsets_from_result)
    from pyannow.ion_channels.celegans_hh import DEFAULT_PARAMS
    from pyannow.step1_svd.encoder import rsvd, neural_scores, choose_k_by_variance
    from pyannow.step1_svd.procrustes import procrustes_align, build_chopin_features
    from pyannow.step2_clustering.motor_primitives import (
        pca_reduce, find_motor_primitives, cluster_to_notes, choose_k_silhouette)
    from pyannow.step3_regression.ridge_composer import RidgeComposer
    from scipy.signal import find_peaks

    rng = np.random.default_rng(seed)

    # ── 1. Worm forward model ────────────────────────────────────────────
    result_worm = run_forward_fast(
        DEFAULT_PARAMS, duration_s=duration_s, dt_ms=0.5,
        drive_freq_hz=1.5, drive_amplitude=12.0, random_seed=42)
    V_mus  = result_worm['V_muscles']
    t_ms   = result_worm['t_arr_ms']
    t_arr  = t_ms * 1e-3
    T_pts  = len(t_arr)
    n_muscles = V_mus.shape[1]

    # 302-neuron synthetic matrix
    n_neurons = 302
    X_neural = np.zeros((n_neurons, T_pts))
    for i in range(n_neurons):
        muscle_idx = i % n_muscles
        noise_amp  = 0.05 * (1 + i / n_neurons)
        X_neural[i] = V_mus[:, muscle_idx] + rng.standard_normal(T_pts) * noise_amp

    # Chopin
    midi_path = Path(midi_rel).resolve()
    events_chopin, bpm = parse_midi(midi_path)
    chopin_onsets = note_onsets(events_chopin, clip_s=duration_s)
    k_chopin = 8
    C_chopin = build_chopin_features(
        events_chopin, duration_s=duration_s, n_bins=T_pts, k_chopin=k_chopin)

    # ── Step 0 baseline ──────────────────────────────────────────────────
    onsets_base = np.array([e[0] for e in result_worm['note_onsets_s']])
    onsets_base = onsets_base[onsets_base <= duration_s]

    # ── Step 1: SVD + Procrustes ─────────────────────────────────────────
    k_worm = min(choose_k_by_variance(X_neural, variance_threshold=0.90), 4)
    U_k, s_k, Vt_k = rsvd(X_neural, k=k_worm, q=1, seed=seed)
    Z_worm = neural_scores(X_neural, U_k).T
    result_proc = procrustes_align(Z_worm, C_chopin)
    Z_aligned   = Z_worm @ result_proc['R']
    activ1 = np.abs(Z_aligned).max(axis=1)
    peaks1, _ = find_peaks(activ1, distance=int(0.28/0.5e-3), height=activ1.mean())
    onsets_step1 = t_arr[peaks1]

    # ── Step 2: PCA + KMeans ─────────────────────────────────────────────
    scores, pca_obj = pca_reduce(X_neural, n_components=4, standardize=True)
    sil_res = choose_k_silhouette(scores, k_range=range(2, 7))
    k_clust = sil_res['best_k']
    labels, _ = find_motor_primitives(scores, k=k_clust)
    onsets_step2 = np.array([e[0] for e in cluster_to_notes(labels, t_ms, min_interval_ms=280)])
    onsets_step2 = onsets_step2[onsets_step2 <= duration_s]

    # ── Step 3: Ridge ────────────────────────────────────────────────────
    rc = RidgeComposer(alpha=None)
    rc.fit(Z_worm, C_chopin)
    C_pred_ridge = rc.predict(Z_worm)
    activ3 = np.abs(C_pred_ridge).max(axis=1)
    peaks3, _ = find_peaks(activ3, distance=int(0.28/0.5e-3), height=activ3.mean())
    onsets_step3 = t_arr[peaks3]

    # We deliberately skip Steps 4-6 (MLP + Adam + L-BFGS) and Step 8 (PINN)
    # in the loader to keep the cache build fast.  They can be added later
    # by reusing the patterns from notebook 03 cell 16 and 18.

    out = {
        "t_arr":         t_arr,
        "X_neural":      X_neural,
        "C_chopin":      C_chopin,
        "chopin_onsets": chopin_onsets,
        "labels_step2":  labels,
        "pca_scores":    scores,
        "Z_worm":        Z_worm,
        "Z_aligned":     Z_aligned,
        "steps": [
            StepOutputs("Step 0 (baseline)",       onsets_base,  None,    len(onsets_base)),
            StepOutputs("Step 1 (SVD+Procrustes)", onsets_step1, activ1,  len(onsets_step1)),
            StepOutputs("Step 2 (PCA+KMeans)",     onsets_step2, None,    len(onsets_step2)),
            StepOutputs("Step 3 (RidgeCV)",        onsets_step3, activ3,  len(onsets_step3)),
        ],
        "meta": {
            "duration_s": float(duration_s),
            "n_neurons":  int(n_neurons),
            "k_worm":     int(k_worm),
            "k_chopin":   int(k_chopin),
            "seed":       int(seed),
        },
    }
    _save_cache(out, cache_path)
    return out


def _save_cache(out: dict, path: Path) -> None:
    """Persist `out` to a .npz file (lossless, no pickle)."""
    steps = out["steps"]
    payload = {
        "t_arr":         out["t_arr"],
        "X_neural":      out["X_neural"],
        "C_chopin":      out["C_chopin"],
        "chopin_onsets": out["chopin_onsets"],
        "labels_step2":  out["labels_step2"],
        "pca_scores":    out["pca_scores"],
        "Z_worm":        out["Z_worm"],
        "Z_aligned":     out["Z_aligned"],
        "step_names":    np.array([s.name for s in steps], dtype=object),
    }
    for i, s in enumerate(steps):
        payload[f"onsets_step{i}"] = s.onsets
        if s.activation is not None:
            payload[f"activ_step{i}"] = s.activation
        payload[f"n_notes_step{i}"] = np.array([s.n_notes])
    np.savez_compressed(path, **payload, meta=json.dumps(out["meta"]))


def _load_cache(path: Path) -> dict:
    """Reverse of _save_cache."""
    with np.load(path, allow_pickle=True) as z:
        names = z["step_names"]
        steps = []
        for i, name in enumerate(names):
            activ_key = f"activ_step{i}"
            activ = z[activ_key] if activ_key in z.files else None
            steps.append(StepOutputs(
                name=str(name),
                onsets=z[f"onsets_step{i}"],
                activation=activ,
                n_notes=int(z[f"n_notes_step{i}"][0]),
            ))
        return {
            "t_arr":         z["t_arr"],
            "X_neural":      z["X_neural"],
            "C_chopin":      z["C_chopin"],
            "chopin_onsets": z["chopin_onsets"],
            "labels_step2":  z["labels_step2"],
            "pca_scores":    z["pca_scores"],
            "Z_worm":        z["Z_worm"],
            "Z_aligned":     z["Z_aligned"],
            "steps":         steps,
            "meta":          json.loads(str(z["meta"])),
        }
