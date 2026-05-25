#!/usr/bin/env python3
"""Create a worm ingestion manifest and linear mapping weights from motif bank.

- Searches for a pattern bank (primary or fallback).
- Computes per-motif centroid features (z-scored features).
- Builds a simple linear mapping (least squares) from motif features -> control gains.
- Saves weights and a manifest to `step_outputs/v2_patterns/dance_archive/worm_ingest/`.

This script does not require scikit-learn or matplotlib.
"""
import json
import sys
from pathlib import Path
import numpy as np

# find repo root
def find_repo_root(start=None):
    start = Path.cwd() if start is None else Path(start)
    for candidate in (start, *start.parents):
        if (candidate / 'PyANNOW').is_dir() and (candidate / 'shared').is_dir():
            return candidate
    raise RuntimeError('Repo root not found')

REPO_ROOT = find_repo_root()
OUTDIR = REPO_ROOT / 'PyANNOW' / 'notebooks' / 'step_outputs' / 'v2_patterns' / 'dance_archive'
OUTDIR.mkdir(parents=True, exist_ok=True)

# candidate pattern bank files
candidates = [
    OUTDIR / 'chopin_pattern_bank.npz',
    OUTDIR / 'chopin_pattern_bank_fallback.npz',
]

pattern_path = None
for p in candidates:
    if p.exists():
        pattern_path = p
        break

if pattern_path is None:
    print('No pattern bank found in', OUTDIR)
    print('Run the pattern builder first, e.g. scripts/build_pattern_bank_fallback.py or execute the notebook cells.')
    sys.exit(2)

print('Using pattern bank:', pattern_path)
data = np.load(pattern_path, allow_pickle=True)
# required arrays: window_features_z OR window_features
if 'window_features_z' in data:
    Xz = data['window_features_z']
else:
    X = data['window_features']
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

if 'motif_representatives' in data:
    reps = np.asarray(data['motif_representatives'], dtype=int)
    # derive labels by nearest rep
    rep_vectors = Xz[reps]
    dmat = np.linalg.norm(Xz[:, None, :] - rep_vectors[None, :, :], axis=2)
    labels = np.argmin(dmat, axis=1)
else:
    labels = data.get('motif_labels')
    if labels is None:
        # fallback: kmeans-like split via simple quantiles
        n_motifs = min(6, max(3, len(Xz) // 120))
        labels = (np.argsort(Xz[:, 0]) % n_motifs)
        reps = []
        rep_vectors = np.stack([Xz[labels == i].mean(axis=0) for i in range(n_motifs)])

n_motifs = int(np.max(labels) + 1)
print('Detected motifs:', n_motifs)

# compute centroid features per motif
centroids = np.zeros((n_motifs, Xz.shape[1]), dtype=float)
counts = np.zeros(n_motifs, dtype=int)
for i in range(n_motifs):
    mask = labels == i
    if mask.sum() == 0:
        centroids[i] = np.zeros(Xz.shape[1])
    else:
        centroids[i] = Xz[mask].mean(axis=0)
    counts[i] = mask.sum()

# Build control target vectors using heuristics (same semantics as notebook):
# control = [command_gain, motor_gain, muscle_gain, phase_shift, refractory_bias]
# We'll use columns: 0=active_mean,1=onset_rate,2=pitch_span,3=pitch_entropy,4=transition_energy,5=pitch_center
if 'saliency' in data:
    # compute motif saliency as mean saliency of assigned windows
    sal = data['saliency']
    motif_sal = np.array([sal[labels == i].mean() if (labels == i).any() else 0.0 for i in range(n_motifs)])
else:
    motif_sal = centroids[:, 1] * 0.8 + centroids[:, 0] * 0.2

feat_onset = centroids[:, 1]
feat_mean = centroids[:, 0]
feat_span = centroids[:, 2]
feat_center = centroids[:, 5]
feat_energy = centroids[:, 4]

command_gain = np.clip(0.35 + 0.18 * motif_sal, 0.0, 1.5)
motor_gain = np.clip(0.30 + 0.22 * feat_onset, 0.0, 1.8)
muscle_gain = np.clip(0.25 + 0.06 * feat_span, 0.0, 2.0)
phase_shift = np.tanh((feat_center - (data['pitches'].shape[0] / 2.0)) / 10.0)
refractory_bias = np.clip(1.2 - 0.15 * feat_energy, 0.0, 1.2)

Y = np.stack([command_gain, motor_gain, muscle_gain, phase_shift, refractory_bias], axis=1)
X_design = np.concatenate([centroids, np.ones((n_motifs, 1))], axis=1)  # add bias

# fit linear mapping W so that X_design @ W ≈ Y
W, residuals, rank, svals = np.linalg.lstsq(X_design, Y, rcond=None)
print('Linear mapping fitted. residuals sum:', float(np.sum(residuals)) if len(residuals) else 0.0)

# save mapping
ingest_dir = OUTDIR / 'worm_ingest'
ingest_dir.mkdir(parents=True, exist_ok=True)
weights_path = ingest_dir / 'worm_ingest_weights.npz'
manifest_path = ingest_dir / 'worm_ingest_manifest.json'
np.savez_compressed(weights_path, W=W, feature_mean=centroids, counts=counts)

manifest = {
    'pattern_bank': str(pattern_path),
    'n_motifs': int(n_motifs),
    'weights': str(weights_path),
    'feature_names': ['z_feat_%d' % i for i in range(Xz.shape[1])],
    'control_names': ['command_gain', 'motor_gain', 'muscle_gain', 'phase_shift', 'refractory_bias'],
    'notes': 'Linear ridge-less least-squares mapping from motif centroid features to worm control gains',
}
with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)

print('Wrote weights ->', weights_path)
print('Wrote manifest ->', manifest_path)
print('Done.')
