# Ion channels — the centerpiece

> The whole project pivots on this: **ion channels tune the worm's neural firing, the firing triggers the piano, and the piano makes music.** A Physics-Informed Neural Network (PINN) learns the channel kinetics that make the worm musical.

---

## Why this is the centerpiece

The user has prior academic work in this exact area:

- **`channelworm-OW.pdf`** — research proposal for ChannelWorm-style modelling (part of the AppStat 2026 project).
- **SC-PINN** ([naml-ion-channel-pinn](../Numerical_analysis_forML/naml-ion-channel-pinn/)) — Structure-Conditioned PINN that predicts ion-channel kinetics `(m_∞, h_∞, τ_m, τ_h)` from structural features. Three frameworks compared: Markov 3-state baseline, HH-PINN, **SC-PINN**.

Wormuse reuses the **ideas** of SC-PINN (not the code directly — this repo is self-contained per design decision) inside `PyANNOW/src/pyannow/ion_channels/`. It becomes the bridge between **biophysics** and **music**.

## The biophysical chain

```
ion-channel kinetics       ←  PINN-learned, parameterized by (V, structural latent z)
        ↓
gating variables m, h      ←  HH gating equations
        ↓
membrane voltage V(t)      ←  per-neuron Hodgkin-Huxley ODE
        ↓
spike events               ←  threshold crossing
        ↓
muscle activations         ←  synaptic projection (C302 connectome)
        ↓
worm pose + body waves     ←  Sibernetic SPH solver
        ↓
piano-trigger schedule     ←  neuron_to_midi/ policy
        ↓
piano audio                ←  piano_sim/ FEM model
```

Change the PINN's prediction of `(m_∞, h_∞, τ_m, τ_h)` → entire chain shifts → the music shifts.

## What the PINN learns

For a *C. elegans* neuron i at voltage V:

$$
\frac{dm_i}{dt} = \frac{m_{\infty,i}(V) - m_i}{\tau_{m,i}(V)},
\qquad
\frac{dh_i}{dt} = \frac{h_{\infty,i}(V) - h_i}{\tau_{h,i}(V)}.
$$

The PINN approximates the four scalar functions `(m_∞(V), h_∞(V), τ_m(V), τ_h(V))` with a small Flax MLP, trained to minimize:

$$
\mathcal{L} =
\underbrace{\sum_i (m_i^{\text{pred}} - m_i^{\text{data}})^2}_{\text{data fit}}
+ \lambda_{\text{phys}}
\underbrace{\int \left\| \frac{dm}{dt}_{\text{pred}} - \frac{m_\infty - m}{\tau_m} \right\|^2 dt}_{\text{HH residual}}.
$$

The HH-residual term is computed via `jax.grad` at random collocation points — no FEM mesh, no FD stencil. This is **exactly** the PINN recipe from NAML L27.

## What the UI exposes

The static UI ships with ≥ 10 pre-rendered scenarios at different points in PINN parameter space:

| Slider | Effect |
|---|---|
| `τ_m` (sodium activation) | Faster `τ_m` → faster spike rise → brighter, more staccato notes |
| `τ_h` (sodium inactivation) | Faster `τ_h` → shorter spikes → fewer sustained notes |
| `V_thresh` (firing threshold) | Lower threshold → more spikes → denser musical texture |
| `g_K` (potassium conductance) | Stronger K current → quicker repolarization → cleaner musical phrasing |

A user moves a slider; the UI swaps in a pre-rendered scenario JSON and plays the corresponding MIDI via Web Audio. **Live tweak** is supported in the JupyterLite UI for a smaller surrogate model.

## Course mapping

| Course | Lecture | Where in PyANNOW |
|---|---|---|
| NAML L14 (autodiff) | computing `∂m/∂V` and `∂m/∂t` | `ion_channels/pinn.py` |
| NAML L20 (Adam variants) | first-stage optimizer | `training/adam_stage.py` |
| NAML L22 (L-BFGS) | second-stage polish | `training/lbfgs_stage.py` |
| NAML L24 (universal approximation) | justifying the MLP shape choice | docstring + `docs/design_notes/` |
| NAML L27 (PINNs) | the entire architecture | `ion_channels/pinn.py` |
| AppStat L01 (PCA) | reducing the structural latent z | `wormuse-analytics/Lab_II_PCA.ipynb` |
| AppStat L04 (LM diagnostics) | regressing music quality on `(τ_m, τ_h, …)` | `wormuse-analytics/Lab_V_regression.ipynb` |

## How this differs from polimuse

| Aspect | polimuse | wormuse |
|---|---|---|
| Composer | A robot learning RL on piano | The worm itself, via its neurons |
| ML core | RL + auxiliary physics losses | PINN for biophysics + a generative composer |
| Biological grounding | None | C. elegans, the only animal with a fully-mapped connectome |
| Tuning knobs | Robot policy weights | Ion-channel parameters |
| Academic anchor | Curriculum learning + piano physics | ChannelWorm + the user's SC-PINN |

## Outputs that prove the PINN matters

By the end of Phase 6 we should be able to demonstrate:

1. **Sweep**: vary `τ_m` from 0.1 ms → 5 ms in 10 steps, render audio at each. Plot a spectrogram — the dominant frequencies shift smoothly.
2. **Ablation**: replace the PINN with a constant-kinetics baseline. Show via the AppStat classifier (Lab VI) that the PINN-driven melodies are statistically more "musical" (higher AUC against a labelled set).
3. **Inverse problem**: given a target spectrogram, run gradient descent through the PINN to find ion-channel parameters that approximate it. This is the **inverse-problem use of PINNs** (NAML L27).

## What is NOT a PINN here

Just for clarity (the term is sometimes overloaded):

- The **composer** (`PyANNOW/src/pyannow/composer/`) is a vanilla sequence model — **not** a PINN. It learns from data, no physics residual.
- The **encoder** (`PyANNOW/src/pyannow/neural_state/`) is a vanilla autoencoder — not a PINN.
- Only the **ion-channel module** is a PINN. It is the seed; everything else hangs off it.
