# PyANNOW

**The NAML sub-project.** Python / JAX / Flax / Optax composer that tunes the worm's nervous system so each movement produces a coherent melody. The **ion-channel PINN is the centerpiece** (see [`../ION_CHANNELS.md`](../ION_CHANNELS.md)).

`PyANNOW` ≈ **Py**thon **A**rtificial **N**eural-channel **N**etwork **O**rchestrator for **W**ormuse.

---

## Layout

```
PyANNOW/
├── pyproject.toml
├── src/pyannow/
│   ├── ion_channels/      The PINN: HH kinetics learned with physics-residual loss
│   ├── neural_state/      Encoder: 302-D neural trajectory → low-dim latent
│   ├── composer/          Latent → MIDI sequence (Flax MLP / seq model)
│   ├── physics_loss/      Couples composer back to ion-channel dynamics
│   └── training/          Adam → L-BFGS pipeline (course-canonical)
├── notebooks/
│   ├── 01_ion_channels_pinn.ipynb       Train PINN on synthetic HH
│   ├── 02_worm_state_encoder.ipynb      Autoencode 302-D trajectories
│   ├── 03_composer_training.ipynb       Train the composer
│   └── 04_end_to_end_inference.ipynb    Worm → music end-to-end
└── tests/                 pytest unit tests
```

## Stack

Per the NAML labs (Lab04-10 are JAX-based; Lab10 introduces Flax + Optax):

```bash
pip install jax jaxlib flax optax \
            numpy scipy matplotlib seaborn \
            pandas scikit-learn \
            jupyter
# Optional GPU:
# pip install "jax[cuda12]"
```

Python 3.10-3.13.

## Lecture map (NAML)

### Linear algebra & low-rank (NAML L02–L09)

| Lecture | Concept | Where |
|---|---|---|
| L02-05 Linear algebra | Inner products, projection, orthogonality | `neural_state/encoder.py` (whitening) |
| L06 Eckart-Young proof | Best low-rank approximation = truncated SVD | `neural_state/svd_encoder.py` (baseline encoder) |
| L08 PCA | PCA of 302-D neural states for visualization | `neural_state/diagnostics.py` |
| L09 Pseudoinverse | Least-squares decoder from latent | `neural_state/decoder.py` |

### Optimization (NAML L18–L22)

| Lecture | Concept | Where |
|---|---|---|
| L18 GD | Plain SGD baseline for comparison | `training/sgd.py` |
| L19-20 SGD + Adam | `optax.adam` first-stage | `training/adam_stage.py` |
| L21 Newton | (Reference only — Hessian too large for these models) | `docs/design_notes/newton_vs_lbfgs.md` |
| L22 L-BFGS | `optax.lbfgs` polish stage (the PINN literature's standard recipe) | `training/lbfgs_stage.py` |

### Autodiff & NNs (NAML L14–L17, L23–L26)

| Lecture | Concept | Where |
|---|---|---|
| L14 Autodiff | `jax.grad` (reverse mode), `jax.jvp`/`vjp` for higher-order | `ion_channels/pinn.py` |
| L15 Activations | tanh for PINNs (smoother gradients than ReLU at boundaries) | `ion_channels/pinn.py` |
| L16-17 Neural networks | Flax `nn.Dense` + `tanh`, Xavier init | `ion_channels/pinn.py` |
| L23 Convolution | (Optional) 1D conv composer for time-series MIDI | `composer/conv_composer.py` (Phase 6+) |
| L24 Universal approximation | Justification for MLP width/depth choice | `docs/design_notes/architecture.md` |
| L25 Functional analysis | Sobolev-style smoothness of `m_∞`, `τ_m` priors | `docs/math_derivations/regularity.md` |
| L26 NN complexity | Parameter count vs training-set size budget | `docs/design_notes/capacity.md` |

### PINNs (NAML L27 — **the core**)

| Lecture | Concept | Where |
|---|---|---|
| L27 PINNs | Physics-informed loss: data + residual + boundary | `ion_channels/pinn.py` + `physics_loss/hh_residual.py` |
| L27 (continued) | Adaptive collocation point sampling (RAR) | `training/rar.py` |
| L27 (continued) | Inverse problem: gradient through PINN to recover params | `notebooks/04_end_to_end_inference.ipynb` |

## Phases

- **Phase 3** — `ion_channels/` PINN trained on synthetic HH. Adam→L-BFGS pipeline working end-to-end.
- **Phase 6** — full composer chain: PINN → encoded states → MIDI. Three audibly distinct scenarios from three ion-channel parameter sets.
- **Phase 8** — (stretch) inverse problem: given a target spectrogram, solve for ion-channel parameters.

See [../ROADMAP.md](../ROADMAP.md) for the full plan.

## Conventions

- **JAX-functional style** — pure functions, immutable arrays, explicit RNG via `jax.random.PRNGKey`.
- `@jax.jit` every hot inner loop. Profile with `jax.profiler`.
- Use **Flax `nn.Module`** for any model with parameters; **plain functions** for stateless math.
- Optimizer state via `optax` triples (`init`, `update`, `apply_updates`).
- All matplotlib styling matches the NAML labs (matter-of-fact, no fancy themes).

## When stuck

- `polimi-naml` skill: `svd-and-rsvd.md`, `pca-and-projections.md`, `optimization-methods.md`, `neural-networks.md`, `pinns.md`.
- The 5 NAML code snippets (svd compression, RSVD, PCA from scratch, optimization zoo, PINN-1D-Poisson) are working references.
- `polimi-appstat` skill for statistical diagnostics of training curves and outputs.
- The user's `naml-ion-channel-pinn/` (in `Numerical_analysis_forML/`) is the **conceptual** reference — read its README and notebooks for the architecture rationale. We do **not** depend on it as a library (per design).
