# wormuse — Open Issues & TODO

Tracked here: correctness bugs, model improvements, sound quality, and stretch goals.
Each item cites the relevant course skill and ROADMAP phase.

---

## 🐛 BUG — Piece is misidentified

| Field | Value |
|---|---|
| **Severity** | Medium (affects all documentation and analysis) |
| **File** | `shared/examples/frederic-chopin-nocturne-no20.mid` |
| **Symptom** | The MIDI filename says "Nocturne No. 20" but the internal track metadata reads `Prelude No. 15 in Db - the Raindrop, Frederic Chopin` |
| **Reality** | The file contains **Chopin's Prelude in D♭ major, Op. 28 No. 15** ("Raindrop"), *not* Nocturne No. 20 (Op. posth., C# minor). These are completely different pieces. |

### Fix

- [ ] Rename the file to `frederic-chopin-prelude-no15-raindrop.mid` and update all references
- [ ] If Nocturne No. 20 (C# minor, posthumous) is actually desired: download the correct MIDI
- [ ] Update the Chopin notebook, living doc, and presentation to use the correct title
- [ ] Note: the **key signature** in the MIDI is D♭ major, which matches the Raindrop Prelude — so the pentatonic pitch mapping (D♭ major pentatonic) was actually correct for the wrong reason

---

## 🐛 BUG — Biological ceiling is computed pessimistically (wrong formula)

| Field | Value |
|---|---|
| **Severity** | High (the "57.7% ceiling" headline is misleading) |
| **File** | `PyANNOW/src/pyannow/targets/midi_target.py` — `biological_ceiling()` |
| **Symptom** | The function treats ALL Chopin notes as coming from ONE voice and checks if consecutive notes are ≥ τ_refrac apart. This ignores the 8 independent muscle groups. |

### Why it's wrong

The current algorithm:
```
reachable = 1; last = t_clip[0]
for t in t_clip[1:]:
    if t - last >= tau_refrac: reachable += 1; last = t
```

This is the **single-voice reachability**. With one voice at τ_refrac = 65 ms → max rate = 15 Hz. But Chopin averages only 2.67 notes/s total. Since we have **8 independent voices**, the correct capacity is `8 × 15 Hz = 120 notes/s >> 2.67 notes/s`. The true note-rate ceiling for 8 voices is **not the bottleneck at all**.

The real ceiling is **rhythmic regularity**: the worm fires on a regular body-wave grid; Chopin is syncopated and irregular.

### Fix

- [ ] Replace `biological_ceiling()` with a **multi-voice greedy scheduler**:
  ```python
  def biological_ceiling_nvoices(params, target_onsets, n_voices=8, window_s=30):
      # Greedy: assign each target note to the first available voice
      last_fired = [-inf] * n_voices
      reachable = 0
      for t in sorted(target_onsets[target_onsets <= window_s]):
          for v in range(n_voices):
              if t - last_fired[v] >= tau_refrac:
                  last_fired[v] = t
                  reachable += 1
                  break
      return reachable / len(target_onsets[target_onsets <= window_s])
  ```
- [ ] Re-run all ceiling analyses in `02_chopin_worm_optimizer.ipynb` and `03_pyannow_naml_progression.ipynb`
- [ ] Update the presentation "biological limits" slide with the corrected numbers
- [ ] Expected outcome: the n=8 voice ceiling should be **much higher** (likely >90%) for Chopin's note density, shifting the story from "57.7% ceiling" to "the real limit is rhythmic regularity, not note rate"

---

## 🚧 IMPROVEMENT — Break the "58%" ceiling: new assumptions & ideas

Even after fixing the ceiling calculation, several hard limits remain. This section lists ideas for genuinely pushing the worm closer to Chopin.

### Idea 1 — Use all 95 BWM cells, not just 8 *(polimi-amsc + polimi-appstat)*

C. elegans has **95 body-wall muscle cells** (not 8). The 8-group model is an oversimplification. With 95 voices:
- 24 dorsal anterior, 23 dorsal posterior, 24 ventral anterior, 24 ventral posterior
- Each can fire independently with its own timing
- Total polyphony: 95 simultaneous notes (more than any piano piece needs)
- Implementation: expand `MUSCLE_PITCHES` from 8 to 95; map to a chromatic scale or the full Chopin pitch range (MIDI 32-99)
- Task: update `worm_optimizer_fast.py` to handle N_muscles as a parameter

**Expected impact:** full chromatic pitch range available; polyphony up to ~10 simultaneous notes (matching Chopin's chords)

### Idea 2 — Use dorsal/ventral separately (2× note density) *(polimi-amsc)*

The worm's dorsal and ventral muscles fire in **antiphase** (dorsal contracts when ventral relaxes and vice versa). This creates two interleaved streams of notes:
- Dorsal stream: fires at phase 0, π/2, π, 3π/2 (every half-cycle)
- Ventral stream: fires at phase π/4, 3π/4, 5π/4, 7π/4 (offset by π/4)
- Net effect: 2× the note rate at the same locomotion frequency
- Already partially modelled in `worm_optimizer_fast.py`'s `muscle_phases` — just needs to be exposed

**Expected impact:** 2× note rate → better coverage of fast passages in the Prelude

### Idea 3 — Reduce τ_Ca via ion-channel optimization *(PyANNOW PINN)*

The refractory period is τ_refrac = τ_Ca + 50 ms. With the default τ_Ca = 15 ms → τ_refrac = 65 ms.
But EGL-19 in *C. elegans* can have τ_Ca as short as **2-5 ms** in some cell types (fast-activating splice variant).
Reducing τ_Ca from 15 ms to 3 ms would give τ_refrac = 53 ms → marginally better.
More importantly: the **50 ms passive relax term** can also be reduced by increasing g_EXP2 (faster K⁺ repolarisation).

- [ ] Add τ_relax as a tunable parameter alongside τ_Ca
- [ ] Compute ceiling as `tau_refrac = tau_Ca + tau_relax` where `tau_relax` depends on `g_EXP2`
- [ ] PINN should optimise both simultaneously

### Idea 4 — Accept chords (2+ simultaneous notes) *(PyANNOW step4-6)*

Current mapping: one muscle → one note, sequential. But the worm's 8 muscle groups **all fire in the same locomotion cycle** (just at different phases). If two muscles fire within < 30 ms of each other, they sound as a chord.
- The Raindrop Prelude is famous for its persistent A♭/G# "raindrop" motif — a repeated note in the same voice
- The left-hand alberti bass can be approximated by alternating dorsal/ventral
- Task: change the MIDI rendering to treat notes within 30 ms as a chord

### Idea 5 — Time-stretch Chopin to match the worm's tempo *(entirely valid musically)*

If the worm's natural rhythm is 0.4 Hz, it plays at ~0.4/2.67 × Chopin's tempo = **15% of Chopin's speed**. But the Raindrop Prelude has famously been played very slowly by some performers (Sviatoslav Richter, ~75% of typical tempo). A tempo-stretched version at 30-40% speed is:
- Musically legitimate (tempo rubato)
- Biologically faithful (uses the worm's actual locomotion speed)
- 100% reachable (every note can be played — the worm has enough speed)

- [ ] Implement `time_stretch_midi(events, factor)` utility
- [ ] Generate a 0.15× speed version of the Raindrop Prelude for baseline comparison
- [ ] Show in the notebook: "at its own tempo, the worm plays 100% of notes"

### Idea 6 — Polyphony via concurrent body segments *(polimi-nla connection)*

The 302-neuron connectome drives muscles in wave-like synchrony. Use **NLA-style SVD** to find the top-k independent **spatial modes** of muscle activation (not just 8 sequential groups). Each mode fires at a distinct frequency:
- Mode 1 (forward crawl): 0.4 Hz, activates dorsal anterior first
- Mode 2 (turning): ~0.1 Hz, activates asymmetrically
- Mode 3 (reversal): ~0.3 Hz, activates posterior first

Map each mode to a different harmonic layer (bass / melody / treble). This creates natural polyphony.

---

## 🔊 IMPROVEMENT — More realistic piano sound synthesis

The current `piano_synth.py` uses **modal synthesis** (40 decaying sine waves). This sounds like a "tin can" or digital marimba, not a real piano, because it lacks:
- String-to-string sympathetic resonance
- Soundboard colouration
- Multi-string unison detuning (real pianos have 2-3 strings per note)
- Hammer mechanical noise (the "thunk")
- Pedal sustain resonance

### Fix A — FluidSynth + SoundFont (easiest, most realistic) ⭐

```bash
brew install fluid-synth
pip install midi2audio   # Python wrapper for FluidSynth
# Download a free Steinway .sf2: https://musescore.org/en/handbook/soundfonts-and-sfz-files
```

```python
from midi2audio import FluidSynth
fs = FluidSynth(sound_font='/path/to/steinway.sf2')
fs.midi_to_audio('output.mid', 'output.wav')
```

- [ ] Add `fluidsynth` + a good `.sf2` soundfont to `wormuse-analytics/pyproject.toml`
- [ ] Add `sfz_synth.py` module to `PyANNOW/src/pyannow/composer/` that wraps FluidSynth
- [ ] Fall back to modal synthesis if FluidSynth is not installed
- [ ] Recommended free soundfonts: Salamander Grand Piano (CC license), GeneralUser GS

### Fix B — Better modal synthesis (no extra deps)

If FluidSynth is unavailable, improve the modal synthesiser:

| Feature | Current | Fix |
|---|---|---|
| Strings per note | 1 | Add 2-3 slightly detuned strings (chorus effect) |
| Attack noise | None | Add a short white-noise burst at onset (4 ms, -20 dB) |
| Sustain pedal | None | Add residual energy from previous notes that decays slowly |
| Soundboard | None | Convolve output with a short room impulse response (RIR) |
| Inharmonicity B | Fixed 4×10⁻⁴ | Use frequency-dependent B: lower for bass, higher for treble |

- [ ] Implement `render_string_v2()` with 3-string detuning in `piano_synth.py`
- [ ] Add a simple convolution reverb (scipy.signal.fftconvolve with a short RIR)
- [ ] The Chabassier papers in `polimuse-docs/` are the reference for the full model

### Fix C — Real piano samples *(stretch, large files)*

Use pre-recorded piano samples (note-per-key, velocity-layered). The Salamander Grand Piano sample library is ~700 MB but provides perfect realism. Would require adding audio file loading (soundfile/librosa).

---

## 🎵 IMPROVEMENT — Render the full piece (not just 15/227 seconds)

| Current | Target |
|---|---|
| 15s clips in `synthesise_melody()` | Full 227s (3:47) |
| Small WAV files (~650 KB) | Full-length WAV (~25 MB at 22 kHz) |
| 40 notes from Chopin | All 1000 notes |

- [ ] Remove the `clip_s=15.0` default in `synthesise_melody()` — let it render the full duration
- [ ] Add a `max_notes` guard (current: 40 default) — remove or raise to `None`
- [ ] Add a `batch_render()` function that renders long pieces in chunks (avoid OOM for very long pieces)
- [ ] Time the full render: with modal synthesis, 1000 notes × 0.03s/note = ~30s on CPU. With FluidSynth, ~5s.
- [ ] Generate `demo_outputs/full_chopin_raindrop.wav` and `demo_outputs/full_worm_melody.wav`
- [ ] Update the Chopin notebook to play the full piece in the §9 audio cell

---

## 📚 IMPROVEMENT — Update polimi-appstat skill for worm analysis

Per the Further Work in `PyANNOW/docs/PyANNOW_NAML_progression.md`:

- [ ] Run worm forward model 500 times with random ion-channel parameters
- [ ] Collect dataset: `(g_EGL19, V_half_Ca, tau_Ca, g_EXP2) → onset_loss`
- [ ] `wormuse-analytics/notebooks/Lab_V_regression.ipynb` — OLS regression with full diagnostics
- [ ] `wormuse-analytics/notebooks/Lab_VI_classification.ipynb` — "musical" vs "non-musical" classifier
- [ ] Use `polimi-appstat` skill: PCA scree, K-means clustering, Ridge, RF feature importance

---

## 🔧 IMPROVEMENT — polimi-amsc: wire in real OpenWorm

Per the Further Work:

- [ ] **Phase 1 (ROADMAP):** implement `wormuse-sim/src/ow_bridge/docker_runner.hpp` (AMSC L05/L09)
- [ ] Pass real spike events from C302 to `PyANNOW/src/pyannow/step1_svd/encoder.py`
- [ ] Replace synthetic `X_neural` in `03_pyannow_naml_progression.ipynb` with real data
- [ ] Expected: k_worm will increase from 1 to 4-8 (real connectome has more structure)

---

## 🗂️ Priority order

| Priority | Issue | Effort | Impact |
|---|---|---|---|
| 🔴 **P0** | Fix piece name (Raindrop, not Nocturne 20) | 30 min | Documentation correctness |
| 🔴 **P0** | Fix ceiling formula (n-voice greedy) | 1 hour | Changes the headline result |
| 🟠 **P1** | FluidSynth realistic piano audio | 2 hours | Dramatically better demos |
| 🟠 **P1** | Full-piece render (all 1000 notes, 227s) | 1 hour | Complete musical demonstration |
| 🟡 **P2** | 95 BWM cells mapping | 1 day | More voices, better pitch coverage |
| 🟡 **P2** | Dorsal/ventral antiphase (2× note density) | 4 hours | Better coverage of fast passages |
| 🟢 **P3** | Time-stretch Chopin to worm tempo | 2 hours | Honest "worm at its own speed" demo |
| 🟢 **P3** | Improved modal synthesis (multi-string, reverb) | 1 day | Better sound without FluidSynth dep |
| 🔵 **P4** | AppStat ion-channel regression dataset | 2 days | Statistical validation |
| 🔵 **P4** | Real OpenWorm C302 spike data | 1 week | Full biological fidelity |
