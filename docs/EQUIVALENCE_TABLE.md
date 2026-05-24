# Cross-System Equivalence Table

> **Three systems, one physics family.**  
> Every constraint in the *C. elegans* biology has a structural counterpart in the
> piano physics and in the NAML learning algorithms used by PyANNOW.
> This table is the scientific backbone of the wormuse project.

See `SCIENTIFIC_FOUNDATION.md` for the mathematical derivations behind each row.
The NAML column maps to specific course lectures and PyANNOW module paths.

---

## Master table

| # | C. elegans biology | Piano physics | NAML / PyANNOW | Course ref | Code |
|---|---|---|---|---|---|
| 1 | **302 neurons** — the entire nervous system; each fires discrete spikes | **88 piano keys** — each string has a distinct frequency; pressed independently | **k-dimensional latent space** — SVD compresses 302-D neural state to k=4-8 principal components | L06 Eckart-Young | `step1_svd/encoder.py` |
| 2 | **96 BWM muscle cells** (Boyle 4×24 model: DL/VL/DR/VR × 24 segments) — the biological output layer; max polyphony | **88 keys × max 10-finger polyphony** — simultaneous voices limited by player hands | **Output dimension `out_dim`** of the composer MLP; k_chopin=96 in `run_forward_fast` | L16 FFNN, L24 UAT | `step4_ffnn/jax_composer.py`, `composer/worm_optimizer_fast.py` |
| 3 | **EGL-19 (L-type Ca²⁺ VGCC)** — all-or-nothing action potential once threshold crossed | **Hammer–string contact** — nonlinear power-law `F = Kδᵖ` triggered above contact boundary | **ReLU/tanh activation** — threshold nonlinearity; zero below `V_half_Ca` | L15 Activations | `ion_channels/celegans_hh.py` |
| 4 | **τ_Ca (EGL-19 activation time, ~2-20 ms)** — sets note attack speed; limits refractory | **Hammer contact duration (~1-4 ms)** — felt compression time; sets initial spectrum | **τ_Ca in `CelegansChannelParams`** — most sensitive PINN parameter (§7 sensitivity) | L27 PINN | `celegans_hh.py` |
| 5 | **g_EGL19 (maximal Ca²⁺ conductance, mS/cm²)** — determines contraction force | **Hammer velocity `v₀` (m/s)** — controls MIDI velocity (loudness) | **`force_to_velocity()` scale factor** — biological amplification | L09 Pseudoinverse | `celegans_hh.py`, `worm_optimizer.py` |
| 6 | **EXP-2 (delayed-rectifier K⁺)** — repolarises muscle; sets note duration | **String damping σ₀ (s⁻¹)** — frequency-independent decay; sets sustain | **g_EXP2 in `CelegansChannelParams`** — controls note release / inter-note gap | L27 PINN | `celegans_hh.py` |
| 7 | **V_half_Ca (EGL-19 half-activation voltage, mV)** — the firing gate | **Key sensitivity threshold** — force required to press a key to sound | **Ca_THRESH in forward model** — decides which muscles fire each cycle | L15 Activations, L27 | `worm_optimizer_fast.py` |
| 8 | **BWM refractory period (~65 ms)** — minimum time between two contractions of same muscle | **String re-strike minimum** — string must partly relax before rehammer sounds natural | **`Ca_THRESH` guard + cycle detection** — prevents double-firing per locomotion cycle | L27 (collocn.) | `midi_target.py` `biological_ceiling()` |
| 9 | **Locomotion frequency (~0.4-2 Hz)** — speed of the body wave; sets musical tempo | **Tempo (BPM × time signature)** — metronome rate of the piece | **`drive_freq_hz`** — master clock of the forward model; determines notes/second | L18 GD (step size) | `worm_optimizer_fast.py` |
| 10 | **NCA-1/2 (Na⁺ leak conductance)** — background depolarisation; sets baseline excitability | **Sustain pedal** — keeps strings resonating; raises the noise floor | **Bias term in MLP** — shifts activation baseline | L16 FFNN | `jax_composer.py` |
| 11 | **4 locomotion quadrants** (DL, DR, VL, VR) — spatial independence; natural polyphony | **4 piano registers** (bass, tenor, alto, treble) — spectral zones with different timbres | **4 PCA components** — top-4 SVD modes of the 302-D neural trajectory | L08 PCA | `step2_clustering/motor_primitives.py` |
| 12 | **Dorsal/ventral antiphase** — D and V fire in opposing half-cycles; ~2× polyphony | **Left hand / right hand** — independent melodic lines, often in counterpoint | **Two interleaved K-means clusters** — FW/BW motor states as separate voices | L10 K-means | `step2_clustering/motor_primitives.py` |
| 13 | **Bending wave propagation** (head → tail) — sequential segmental activation | **Inharmonicity `B = π²ESκ²/(T_sL²)`** — high modes are slightly sharp; sound disperses | **1D convolution kernel** — CNN encoder captures the wave's temporal spread | L23 Convolution | `step8_pinn/locomotion_pinn.py` |
| 14 | **Connectome topology** (302×302 adjacency) — fixed; determines neural computation | **Soundboard normal modes** — fixed geometry determines resonance spectrum | **Gram matrix `Z^T Z`** in ridge regression | L07 Least squares | `step3_regression/ridge_composer.py` |
| 15 | **Muscle force rescaling** (worm force ~μN → piano force ~10 N) | **`force_scale = 50`** — amplification factor in `force_to_velocity()` | **`force_scale` hyperparameter** — the "size rescaling" bridging biology and instrument | — | `worm_optimizer.py` |
| 16 | **HH gating `m_∞(V)` — Boltzmann curve** (smooth S-shape) | **Hammer felt compression `F(δ) = Kδᵖ`** (power law) | **Sigmoid/tanh activation** `σ(Wx + b)` — both are smooth threshold functions | L15 Activations | `jax_composer.py` |
| 17 | **Muscle Ca²⁺ action potential waveform** (fast rise ~τ_Ca, slow decay ~τ_h) | **String envelope** (sharp attack, exponential decay `e^{-σ₀t}`) | **Loss target** for the composer: the temporal envelope of Chopin notes | L27 PINN data loss | `midi_target.py` |
| 18 | **Locomotion ODE** `ÿ + 2γẏ + ω²y = F_neural` (body wave) | **Damped string PDE** `ρü - Tu_xx + σ₀u̇ = F_hammer` | **Physics residual** in Step 8 PINN — the ODE/PDE loss term | L27 PINN | `step8_pinn/locomotion_pinn.py` |
| 19 | **τ_m sweep** (PINN tuning: fast→slow channels) | **Inharmonicity sweep** (vary `B`: bright → warm string) | **Adam → L-BFGS two-stage training** — fast warmup then precise tuning | L20 Adam, L22 L-BFGS | `step5_training/`, `step6_lbfgs/` |
| 20 | **302 neurons → k=4 PCA scores** (Eckart-Young compression) | **N modes → k=40 dominant modes** (modal synthesis truncation) | **Rank-k truncation** `X_k = U_k Σ_k V_k^T` — identical mathematical structure | L06 Eckart-Young | `step1_svd/encoder.py`, `piano_synth.py` |

---

## Summary by correspondence type

### Type A — Threshold / gate (rows 3, 4, 7, 8, 16)
All three systems have a **localised nonlinearity that gates whether anything happens at all**:
- Biology: `V_half_Ca` — below threshold → silence; above → full AP
- Piano: contact depth `δ = 0` — not pressing → silence; pressing → sound
- NAML: activation function threshold — below → zero output; above → non-zero

### Type B — Wave / oscillation (rows 9, 13, 18)
All three systems are **wave-bearing with characteristic frequencies**:
- Biology: bending wave at locomotion frequency `ω_loco`
- Piano: string modes at `f_n = nf₁√(1+Bn²)` (inharmonic)
- NAML: convolutional filters in the CNN encoder (temporal frequency analysis)

### Type C — Decay / damping (rows 6, 17)
All three systems show **exponential energy dissipation after excitation**:
- Biology: muscle relaxation `τ_relax = τ_Ca + 50 ms`
- Piano: string decay `u(t) ∝ e^{-σ₀t}` (damping coefficient)
- NAML: L-BFGS quadratic convergence (fast approach to minimum = fast "settling")

### Type D — Compression / low-rank (rows 1, 11, 20)
All three systems have a **high-dimensional input that projects onto a low-rank output**:
- Biology: 302 neurons → 4 motor-state dimensions (locomotion subspace)
- Piano: infinite string modes → 40 dominant modes (modal synthesis)
- NAML: Eckart-Young theorem — best rank-k approximation is the truncated SVD

---

## Three structural correspondences (SCIENTIFIC_FOUNDATION.md §C.2)

These are the three deepest connections that make the worm→piano mapping
physically justified rather than metaphorical:

| # | Correspondence | Worm | Piano |
|---|---|---|---|
| **C2.1** | Excitable threshold | `V > V_half_Ca` → muscle fires | `δ > 0` → hammer contacts string |
| **C2.2** | Wave-bearing dynamics | Body bending wave at `ω_loco` | String modes at `f_n` (dispersive) |
| **C2.3** | Localised nonlinear forcing | Gating `m_∞(V)` — sigmoidal | Hammer force `F(δ) = Kδᵖ` — power-law |

---

## Version

Added in wormuse `v0.5.0`. Last updated `v0.9.0` (row 2 corrected to 96 BWM cells per Boyle 4×24 model).
Maintained alongside `SCIENTIFIC_FOUNDATION.md`. See `CHANGELOG.md` for version history.
