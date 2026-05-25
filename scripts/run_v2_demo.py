#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import json
import struct
import wave
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = REPO_ROOT / 'PyANNOW' / 'notebooks' / 'step_outputs' / 'v2_patterns'
archive_dir = OUTDIR / 'dance_archive'
ingest_dir = archive_dir / 'worm_ingest'

pattern_bank = archive_dir / 'chopin_pattern_bank.npz'
fallback_bank = archive_dir / 'chopin_pattern_bank_fallback.npz'
weights_path = ingest_dir / 'worm_ingest_weights.npz'

if not weights_path.exists():
    raise SystemExit(f'Worm ingest weights not found at {weights_path}')
if not (pattern_bank.exists() or fallback_bank.exists()):
    raise SystemExit(f'No pattern bank found at {pattern_bank} or {fallback_bank}')

# load weights
W = np.load(weights_path)['W']
print('Loaded W shape:', W.shape)

# load bank
bank_path = pattern_bank if pattern_bank.exists() else fallback_bank
bank = np.load(bank_path, allow_pickle=True)
print('Using pattern bank:', bank_path)

# prepare features z
if 'window_features_z' in bank:
    Xz = bank['window_features_z']
else:
    X = bank['window_features']
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# get centroids
if 'motif_representatives' in bank:
    reps = np.asarray(bank['motif_representatives'], dtype=int)
    centroids = Xz[reps]
else:
    labels = bank.get('motif_labels')
    if labels is None:
        m = min(6, max(3, len(Xz)//120))
        labels = (np.argsort(Xz[:, 0]) % m)
    m = int(labels.max() + 1)
    centroids = np.vstack([Xz[labels == i].mean(axis=0) for i in range(m)])

# pick motif: highest mean saliency if present
if 'saliency' in bank:
    sal = bank['saliency']
    motif_sal = np.array([sal[(bank.get('motif_labels') == i)].mean() if (bank.get('motif_labels') == i).any() else 0.0 for i in range(centroids.shape[0])])
    motif_id = int(np.nanargmax(motif_sal))
else:
    motif_id = 0

feat = centroids[motif_id]
# map
x_design = np.concatenate([feat, np.array([1.0])])
y = x_design @ W
control_names = ['command_gain','motor_gain','muscle_gain','phase_shift','refractory_bias']
ctrl = {name: float(val) for name, val in zip(control_names, y)}
print('Demo motif_id=', motif_id, 'control:', json.dumps(ctrl, indent=2))

# audio
fs = 8000
dur = 4.0
t = np.linspace(0, dur, int(fs * dur), endpoint=False)
freq = 220.0
amp = np.clip(ctrl['command_gain'] * 0.8 + ctrl['motor_gain'] * 0.6, 0.01, 1.0)
carrier = np.sin(2 * np.pi * freq * t)
env = 0.5 * (1.0 + np.sin(2 * np.pi * 1.5 * t))
audio = (carrier * env * amp).astype(float)

out_wav = archive_dir / f'dance_demo_motif_{motif_id}.wav'
# write wav using wave
with wave.open(str(out_wav), 'wb') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(int(fs))
    frames = b''.join(struct.pack('<h', int(max(-32767, min(32767, x * 32767)))) for x in audio)
    w.writeframes(frames)
print('Wrote demo WAV:', out_wav, 'size:', out_wav.stat().st_size)

# controller demo
class SimpleController:
    def __init__(self, W):
        self.W = W
    def control_from_features(self, feat_z):
        x = np.concatenate([np.asarray(feat_z).ravel(), [1.0]])
        return x @ self.W
    def rollout(self, feat_z, duration_s=2.0, fs=50):
        ctrl = self.control_from_features(feat_z)
        n = int(duration_s * fs)
        return np.tile(ctrl[None, :], (n, 1)), np.linspace(0, duration_s, n)

controller = SimpleController(W)
feat_demo = feat
timeline, times_ctrl = controller.rollout(feat_demo, duration_s=4.0, fs=10)
print('Controller rollout shape:', timeline.shape)

demo_out = {'times': times_ctrl.tolist(), 'controls': timeline.round(4).tolist()}
demo_path = ingest_dir / 'controller_demo.json'
with open(demo_path, 'w') as f:
    json.dump(demo_out, f)
print('Saved controller demo JSON to', demo_path, 'size:', demo_path.stat().st_size)

# list archive_dir
print('\nArchive dir listing:')
for p in sorted(archive_dir.iterdir()):
    print('-', p.name, p.stat().st_size)

print('\nWorm ingest dir listing:')
for p in sorted(ingest_dir.iterdir()):
    print('-', p.name, p.stat().st_size)
