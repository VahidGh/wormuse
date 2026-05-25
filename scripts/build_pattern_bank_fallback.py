#!/usr/bin/env python3
"""Build motif bank using numpy-only methods (no sklearn, no matplotlib).
Saves a compressed archive and a manifest JSON.
"""
import sys
from pathlib import Path
import json
import numpy as np

# locate repo root heuristically
def find_repo_root(start=None):
    start = Path.cwd() if start is None else Path(start)
    for candidate in (start, *start.parents):
        if (candidate / 'PyANNOW').is_dir() and (candidate / 'shared').is_dir():
            return candidate
    raise RuntimeError('Repo root not found')

REPO_ROOT = find_repo_root()
sys.path.insert(0, str((REPO_ROOT / 'PyANNOW').resolve()))

try:
    from chopin_score_net.data import build_score_dataset
except Exception as e:
    print('Failed to import chopin_score_net.data:', e)
    raise

MIDI_PATH = REPO_ROOT / 'shared' / 'examples' / 'chopin_nocturne_op_posth_csharp_minor.mid'
OUTDIR = REPO_ROOT / 'PyANNOW' / 'notebooks' / 'step_outputs' / 'v2_patterns'
OUTDIR.mkdir(parents=True, exist_ok=True)

print('MIDI_PATH =', MIDI_PATH)
print('OUTDIR =', OUTDIR)

dataset = build_score_dataset(MIDI_PATH, resolution_s=0.02, n_harmonics=12)
roll = dataset.target_roll.astype(float)
times = dataset.times
pitches = np.asarray(dataset.pitches)
dt = float(times[1] - times[0])

# feature helpers
def shannon_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())

# build pattern bank
def build_pattern_bank(roll, times, dt, win_s=2.0, hop_s=0.10):
    win_bins = max(1, int(round(win_s / dt)))
    hop_bins = max(1, int(round(hop_s / dt)))
    onsets = np.vstack([np.zeros((1, roll.shape[1])), np.clip(np.diff(roll, axis=0), 0, None)])
    rows = []
    windows = []
    for start in range(0, len(roll) - win_bins + 1, hop_bins):
        stop = start + win_bins
        seg = roll[start:stop]
        seg_on = onsets[start:stop]
        pitch_activity = seg.sum(axis=0)
        active_idx = np.flatnonzero(pitch_activity > 0)
        pitch_span = int(active_idx[-1] - active_idx[0] + 1) if len(active_idx) else 0
        onset_rate = float(seg_on.sum() / win_s)
        active_mean = float(seg.sum(axis=1).mean())
        transition_energy = float(np.abs(np.diff(seg, axis=0)).sum())
        pitch_entropy = shannon_entropy(pitch_activity)
        pitch_center = float(np.average(np.arange(len(pitches)), weights=pitch_activity)) if pitch_activity.sum() else 0.0
        rows.append([active_mean, onset_rate, pitch_span, pitch_entropy, transition_energy, pitch_center])
        windows.append((start, stop))
    return np.asarray(rows, dtype=float), np.asarray(windows, dtype=int)

X_pat, windows = build_pattern_bank(roll, times, dt, win_s=2.0, hop_s=0.10)
Xz = (X_pat - X_pat.mean(axis=0)) / (X_pat.std(axis=0) + 1e-8)

saliency = (
    1.3 * Xz[:, 1] +
    1.0 * Xz[:, 0] +
    0.8 * Xz[:, 3] +
    0.7 * Xz[:, 4] -
    0.4 * Xz[:, 2]
)
saliency = (saliency - saliency.mean()) / (saliency.std() + 1e-8)
# smooth
saliency_smooth = np.convolve(saliency, np.ones(5) / 5.0, mode='same')

# choose motif reps by farthest-point sampling among top windows
n_motifs = min(8, max(3, len(Xz) // 90))
top_k = max(200, len(Xz) // 6)
top_idx = np.argsort(saliency_smooth)[-top_k:]
rep_idxs = [int(top_idx[-1])]
for _ in range(1, n_motifs):
    reps = Xz[rep_idxs]
    cand = Xz[top_idx]
    # distances from each candidate to nearest rep
    dists = np.min(np.linalg.norm(cand[:, None, :] - reps[None, :, :], axis=2), axis=1)
    next_local = int(top_idx[np.argmax(dists)])
    rep_idxs.append(next_local)
rep_vectors = Xz[rep_idxs]
# assign labels
dmat = np.linalg.norm(Xz[:, None, :] - rep_vectors[None, :, :], axis=2)
labels = np.argmin(dmat, axis=1)

archive_dir = OUTDIR / 'dance_archive'
archive_dir.mkdir(parents=True, exist_ok=True)
pattern_bank_path = archive_dir / 'chopin_pattern_bank_fallback.npz'
np.savez_compressed(
    pattern_bank_path,
    window_bounds=windows,
    window_features=X_pat,
    window_features_z=Xz,
    saliency=saliency_smooth,
    motif_labels=labels,
    motif_representatives=np.asarray(rep_idxs, dtype=int),
    times=times,
    pitches=pitches,
)

manifest = {
    'version': 'v2.0.0-fallback',
    'n_motifs': int(n_motifs),
    'pattern_bank': str(pattern_bank_path),
}
with open(archive_dir / 'worm_dance_manifest_fallback.json', 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)

print('Saved fallback pattern bank to:', pattern_bank_path)
print('Saved fallback manifest to:', archive_dir / 'worm_dance_manifest_fallback.json')
