# Scientific Foundation

> **Two excitable, wave-bearing systems coupled by a learned map.**
> This document describes the physics on both sides of wormuse: the biological chain (ion channels → spikes → muscle → body) on one side, the mechanical chain (hammer → string → soundboard → air) on the other, and the mapping that ties them together.

---

## Abstract

Wormuse models a closed pipeline in which the electrical activity of a *Caenorhabditis elegans* nervous system, simulated at the resolution of individual ion channels, drives a physically simulated piano. The two systems — the worm and the instrument — are not arbitrary partners: both are **dissipative, nonlinear, excitable systems** governed by partial differential equations on the same family of operators (parabolic in time, hyperbolic in space, with localized nonlinear forcing). A Physics-Informed Neural Network (PINN) learns the ion-channel kinetics; that PINN is the **tuning fork** of the whole pipeline.

This document develops the mathematics of both chains, identifies their structural correspondences, and motivates the spike-to-note mapping as more than a notational trick.

---

## Notation

| Symbol | Meaning | Units |
|---|---|---|
| `V(t)` | membrane voltage of a neuron | mV |
| `m, h, n` | Hodgkin-Huxley gating variables | dimensionless ∈ [0,1] |
| `m_∞(V), τ_m(V)` | equilibrium gating value, time constant | -, ms |
| `g_{Na}, g_K, g_L` | maximal conductances per unit area | mS·cm⁻² |
| `E_{Na}, E_K, E_L` | reversal potentials | mV |
| `C_m` | membrane capacitance per unit area | μF·cm⁻² |
| `I_{ext}` | injected current per unit area | μA·cm⁻² |
| `δ_i(t)` | spike train of neuron `i` (sum of Diracs) | s⁻¹ |
| `a_j(t)` | activation of muscle group `j` | dimensionless ∈ [0,1] |
| `q(t)` | worm body pose (centerline coords) | mm |
| `u(x,t)` | string transverse displacement | mm |
| `w(x,y,t)` | soundboard transverse displacement | mm |
| `p(x,y,z,t)` | acoustic pressure field | Pa |
| `T_s, ρ_s, ES, κ` | string tension, linear density, bending stiffness | N, kg·m⁻¹, N·m², - |
| `D, h, ρ_p` | plate flexural rigidity, thickness, density | N·m, m, kg·m⁻³ |
| `B = π² E S κ² / (T L²)` | string inharmonicity coefficient | - |

---

## Side A — From ion channels to worm movement

### A.1 Ion channel kinetics (the Hodgkin–Huxley model)

Each voltage-gated channel is modelled by gating variables obeying first-order kinetics:

$$
\frac{dm}{dt} = \alpha_m(V)(1 - m) - \beta_m(V) m
\;=\; \frac{m_\infty(V) - m}{\tau_m(V)},
\qquad
m_\infty(V) = \frac{\alpha_m(V)}{\alpha_m(V) + \beta_m(V)},
\quad
\tau_m(V) = \frac{1}{\alpha_m(V) + \beta_m(V)}.
$$

Identical equations hold for `h` (sodium inactivation) and `n` (potassium activation). The voltage-dependent rate functions `α(V), β(V)` are classically empirical (Hodgkin & Huxley 1952); their values determine **how quickly a neuron responds to a stimulus, how long it stays excited, and how fast it can fire again**.

In wormuse, **a PINN learns `(m_∞, h_∞, τ_m, τ_h)` as functions of voltage** (and, in the SC-PINN variant, of a structural latent `z`). The training loss is data-driven plus a physics residual term enforcing the gating ODE — see `ION_CHANNELS.md` for the full derivation.

The musically relevant fact: **`τ_m` and `τ_h` set the time scale of the firing rhythm.** A neuron with `τ_m = 0.1 ms` fires sharp, transient spikes; one with `τ_m = 5 ms` fires broader, slower events. This directly transduces into note attack and tempo in the piano model.

### A.2 Single-neuron dynamics

Sum the ionic currents to get the membrane voltage equation:

$$
C_m \frac{dV}{dt}
\;=\;
-\,g_{Na} \, m^3 h \,(V - E_{Na})
\;-\;g_K \, n^4 \,(V - E_K)
\;-\;g_L \,(V - E_L)
\;+\;I_{ext}(t).
$$

This 4-D nonlinear system (`V, m, h, n`) supports **limit cycles** (regular firing), **excitable thresholds** (a small bump in `I_ext` triggers a full action potential), and **refractoriness** (a fired neuron is briefly insensitive). All three behaviors carry musical meaning:

- Limit cycle → steady rhythm.
- Threshold crossings → discrete note events.
- Refractoriness → notes cannot overlap arbitrarily fast.

A spike event is registered when `V(t)` crosses some threshold `V_θ` from below; the spike train is

$$
\delta_i(t) \;=\; \sum_{k} \delta\bigl(t - t^{(k)}_i\bigr),
\qquad
t^{(k)}_i = \big\{\,t : V_i(t) = V_\theta,\;\dot V_i(t) > 0\,\big\}.
$$

### A.3 Network: the C302 connectome

*C. elegans* has exactly **302 neurons** with a fully mapped connectome (White et al. 1986; Cook et al. 2019). The C302 project simulates this network in NEURON via NeuroML. Each neuron's voltage evolves by A.2; neurons are coupled through:

- **Chemical synapses** — presynaptic spike `δ_j(t)` triggers an exponentially decaying postsynaptic current
  $$
  I_{ij}^{\text{chem}}(t) = g_{ij} \,e^{-(t - t^{(k)}_j)/\tau_{\text{syn}}} \,(V_i - E_{ij}^{\text{syn}}),
  $$
  with reversal potential `E_{ij}^{\text{syn}}` distinguishing excitatory (≈ 0 mV) from inhibitory (≈ -70 mV) connections.

- **Gap junctions (electrical synapses)** — instantaneous ohmic coupling
  $$
  I_{ij}^{\text{gap}}(t) = g^{\text{gap}}_{ij} \,(V_j - V_i).
  $$

The full network is therefore a large coupled ODE system of dimension `4 × 302 = 1208` plus synaptic state. C302 integrates this with stiff ODE solvers; the spike output is what wormuse consumes.

### A.4 Neuromuscular junction (NMJ)

Motor neurons project onto 95 body wall muscle cells. A presynaptic spike triggers acetylcholine release; the postsynaptic membrane depolarizes, and intracellular `[Ca²⁺]` rises. The muscle activation `a_j(t)` is modelled as a first-order filter of the upstream firing rate:

$$
\tau_a \frac{d a_j}{dt} = -a_j + r_j(t),
\qquad
r_j(t) = \sum_{i \in \text{pre}(j)} w_{ij} \int_{-\infty}^{t} \delta_i(s) \, e^{-(t-s)/\tau_{\text{rise}}} ds,
$$

where `w_{ij}` weights the synaptic strength and `τ_a, τ_{rise}` are time constants of activation. The cumulative effect is a smoothed envelope of presynaptic firing.

### A.5 Muscle force generation (Hill-type model)

A standard Hill model expresses muscle force as a product of length, velocity, and activation factors:

$$
F_{\text{muscle},j}(t) \;=\;
a_j(t)\;F_{\max,j}\;f_\ell(\ell_j)\;f_v(\dot\ell_j),
$$

with `f_ℓ(ℓ)` the length-tension relation (Gaussian-like) and `f_v(v)` the force-velocity relation (hyperbolic in concentric, exponential in eccentric). For a worm, `ℓ_j` is the local body length of segment `j` and `\dot ℓ_j` its rate of change.

### A.6 Body mechanics (SPH — Smoothed Particle Hydrodynamics)

The worm body is a soft, near-incompressible fluid embedded in a Newtonian medium. Sibernetic (Palyanov et al. 2018) discretizes the body as a set of particles obeying:

$$
\frac{D \mathbf v_p}{Dt}
\;=\;
-\frac{1}{\rho_p}\nabla P
\;+\;\nu \nabla^2 \mathbf v_p
\;+\;\mathbf f^{\text{muscle}}_p(t)
\;+\;\mathbf g,
$$

with pressure `P` computed from a Tait equation of state. The muscle force `\mathbf f^{\text{muscle}}_p` is the SPH counterpart of the Hill model: each muscle segment exerts a contractile force on the particles in its support, proportional to `F_{muscle, j}(t)` from A.5.

The body pose `q(t)` — a curve through the centerline — is a functional of the entire muscle activation history `{a_j(s) : s ≤ t}`.

### A.7 Putting it together

The chain is hierarchical, each layer a deterministic functional of the previous:

$$
\boxed{\;
(\text{PINN}, V_\theta, \text{connectome})
\;\longmapsto\;
\{δ_i(t)\}_{i=1}^{302}
\;\longmapsto\;
\{a_j(t)\}_{j=1}^{95}
\;\longmapsto\;
q(t)
\;}
$$

Crucially, **only the first arrow contains the PINN-tunable parameters**. Once `{m_∞, τ_m, h_∞, τ_h, …}` are fixed, the rest of the chain is determined by the connectome and the body physics. Music-quality optimization therefore reduces to a low-dimensional search over PINN parameter space — feasible.

---

## Side B — Piano physics

Reference: Chabassier, J., Chaigne, A., Joly, P. *Time Domain Simulation of a Piano. Parts 1 (Model Description) and 2 (Numerical Aspects)*, available in `docs/polimuse-docs/`. Wormuse's piano simulator implements the same physical model, simplified for clarity in Phases 2 and 5.

### B.1 Hammer–string contact

A piano hammer with mass `M_h` strikes the string at speed `v_0` and indents the felt by a depth `δ(t) = u(x_h, t) - z_h(t)`, where `z_h` is the hammer-tip position. The contact force is a power law:

$$
F_h(δ) \;=\; \begin{cases} K_h \,δ^{p_h} & \text{if } δ > 0, \\ 0 & \text{otherwise.} \end{cases}
$$

Empirically `K_h \approx 4 \times 10^9\;\text{N}\cdot\text{m}^{-p_h}` and `p_h \in [2.0, 3.5]` for piano felt. The hammer equation of motion is

$$
M_h \,\ddot z_h \;=\; F_h(δ),
\qquad
z_h(0) = u(x_h, 0),\;\dot z_h(0) = -v_0.
$$

The strike duration is ≈ 1-4 ms — much shorter than the string's free-decay time, so the impulse approximation `F_h(t) \propto δ(t - t_0)` is sometimes used. In wormuse we keep the full nonlinear contact model.

### B.2 Stiff vibrating string

Real piano strings have small but non-negligible bending stiffness. The relevant model is a **stiff, lossy, weakly nonlinear string**:

$$
\rho_s \frac{\partial^2 u}{\partial t^2}
\;-\;T_s \frac{\partial^2 u}{\partial x^2}
\;+\;ES \kappa^2 \frac{\partial^4 u}{\partial x^4}
\;+\;2 \rho_s \sigma_0 \frac{\partial u}{\partial t}
\;-\;2 \rho_s \sigma_1 \frac{\partial^3 u}{\partial t \,\partial x^2}
\;=\;F_h(t)\;\delta(x - x_h),
$$

with boundary conditions `u(0,t) = u(L,t) = 0` (pinned ends) and zero initial conditions. The terms are:

- `T_s ∂²u/∂x²` — restoring force from string tension.
- `ES κ² ∂⁴u/∂x⁴` — bending stiffness (Euler-Bernoulli).
- `2ρ_s σ_0 ∂u/∂t` — frequency-independent damping.
- `2ρ_s σ_1 ∂³u/(∂t ∂x²)` — frequency-dependent damping (high modes decay faster).
- RHS — point-supported hammer force.

The bending term breaks harmonicity: in a perfectly flexible string the mode frequencies are `f_n = n f_1`; with stiffness they become

$$
f_n \;=\; n f_1 \sqrt{1 + B n^2},
\qquad
B \;=\; \frac{\pi^2 E S \kappa^2}{T_s L^2},
$$

which is what gives the piano its characteristic "bright" timbre.

### B.3 Soundboard (Kirchhoff–Love plate)

The bridge transmits string vibration to the soundboard, modelled as a damped Kirchhoff-Love plate:

$$
D \,\nabla^4 w
\;+\;\rho_p h \frac{\partial^2 w}{\partial t^2}
\;+\;\eta_p \frac{\partial w}{\partial t}
\;=\;\sum_n F^{\text{bridge}}_n(t) \,\delta(\mathbf{r} - \mathbf{r}^{\text{bridge}}_n),
$$

where `D = E h³ / (12(1-ν²))` is the flexural rigidity, `ν` is Poisson's ratio, `η_p` is a viscoelastic damping coefficient, and the right-hand side is the sum of point forces from all strings attached to the bridge.

This is a **biharmonic** equation in space (4th order) and **second-order in time**. After multiplying by a test function `v ∈ H²₀(Ω)` and integrating by parts twice, the weak form is

$$
\int_\Omega \rho_p h \,\ddot w \, v
\;+\;
\int_\Omega D \,\Delta w \,\Delta v
\;+\;
\int_\Omega \eta_p \,\dot w \,v
\;=\;
\sum_n F^{\text{bridge}}_n(t) \,v(\mathbf{r}^{\text{bridge}}_n).
$$

This is what wormuse's `wormuse-sim/src/piano_sim/soundboard_fem.cpp` solves in Phase 5 using **deal.II** with continuous `C^1` elements (Hermite triangles) or the equivalent mixed formulation with `H¹` elements + auxiliary variables. Time integration is a `θ`-method (NMPDE Lecture 4) — Crank-Nicolson (θ = ½) is the default, unconditionally stable and second-order.

### B.4 Sound radiation (Helmholtz equation)

The pressure `p(\mathbf{r}, t)` in the surrounding air satisfies the linear acoustic wave equation

$$
\frac{1}{c^2}\frac{\partial^2 p}{\partial t^2} - \nabla^2 p = 0,
\qquad c \approx 343\;\text{m·s}^{-1},
$$

with boundary condition on the soundboard surface `Σ_b`:

$$
\frac{\partial p}{\partial n}\bigg|_{Σ_b} = -\rho_{\text{air}}\,\ddot w.
$$

In the frequency domain this is the Helmholtz equation `(∇² + k²) \hat p = 0` with the Neumann data coming from the soundboard. The radiated sound pressure at the listener position is what we ultimately render to WAV. For the wormuse MVP we skip the explicit Helmholtz solve and use a near-field approximation: the audio signal is proportional to a windowed time-derivative of the soundboard displacement at a chosen "microphone" point.

### B.5 Putting it together (mechanical chain)

$$
\boxed{\;
\text{MIDI event}\;(\text{pitch}, t_0, v_0)
\;\longmapsto\;
F_h(t)
\;\longmapsto\;
u(x,t)
\;\longmapsto\;
w(\mathbf{r}, t)
\;\longmapsto\;
p(\mathbf{r}_{\text{mic}}, t)
\;}
$$

Each arrow is a forward simulation step.

---

## The bridge: from biological spikes to musical notes

### C.1 The mapping function

A *mapping policy* `Φ` converts the spike trains `{δ_i(t)}` into MIDI events. The general form is

$$
\Phi : \{\delta_i\}_{i=1}^{302} \;\longmapsto\;
\Big\{\,(p_k, t_k, v_k)\,\Big\}_{k=1}^{N},
$$

with `p_k` = pitch (MIDI note number), `t_k` = onset time, `v_k` = velocity (loudness).

Wormuse implements three policies, increasing in sophistication:

1. **`one-to-one`** (Phase 2 baseline). Map the first 8 motor neurons to the 8 strings of a C-major scale; each spike → NoteOn at that pitch with `v_k = 64` (mezzo-forte). Time `t_k` = spike time.

2. **`motor-primitive`** (Phase 4-6). Cluster the 302-D spike-rate state into ~10 motor primitives (KMeans + silhouette in `wormuse-analytics/notebooks/Lab_III_clustering.ipynb`). Each cluster centroid is assigned a chord; transitions between clusters trigger chord changes. Velocity is the magnitude of the centroid in PCA space.

3. **`ion-channel-tuned`** (Phase 6+). The PyANNOW composer (Flax MLP) takes the encoded neural latent and emits MIDI sequences directly; the PINN-learned ion-channel kinetics shape the spike statistics that feed the encoder. Music quality is fed back as the regression target in `wormuse-analytics/notebooks/Lab_V_regression.ipynb`.

### C.2 Why this is physically reasonable (not arbitrary)

Three structural correspondences make the mapping more than a notational convenience:

1. **Excitable thresholds on both sides.** A neuron fires when `V` crosses `V_θ`; a piano string is excited when the hammer crosses the felt-contact threshold `δ > 0`. Both systems are **silent below threshold and produce a sharp, finite-energy event above it.** The mapping `spike → NoteOn` is the natural threshold-event homomorphism.

2. **Wave equations at both ends.** The membrane voltage equation (A.2) is a reaction-diffusion equation on the neuron's geometry; if we relax to a cable model with axial diffusion, it becomes a true PDE of telegraph type. The piano string (B.2) is a 1D wave equation with bending. Both support **dispersive wave propagation with characteristic mode frequencies**, and both decay exponentially in time. The matching of timescales — `τ_m, τ_h` for the neuron and `1/(2σ_0)` for the string — is what makes the music sound natural rather than scrambled.

3. **Localized nonlinear forcing.** The HH conductances are sigmoidal functions of `V`, localized in voltage. The hammer-string contact is a power law in `δ`, localized at `x = x_h`. Both systems are **linear background + localized nonlinear input**. This is a clean structural match.

### C.3 Ion channels as musical tuning knobs

The PINN parametrizes `(m_∞, h_∞, τ_m, τ_h)`. Changing these has **predictable, monotone effects on the music**:

| PINN parameter shift | Effect on `δ_i(t)` | Effect on music |
|---|---|---|
| `τ_m` ↓ (sodium activation faster) | sharper spikes, faster onset | brighter attack, staccato |
| `τ_m` ↑ | broader spikes, slower onset | softer attack, legato |
| `τ_h` ↓ (sodium inactivation faster) | shorter spikes, more frequent firing | denser texture, faster tempo |
| `τ_h` ↑ | longer spikes, more refractory | sparser texture, slower tempo |
| `m_∞(V)` curve shifted left | lower firing threshold | more notes per second |
| `m_∞(V)` curve shifted right | higher threshold | rarer, more deliberate notes |
| `g_K` ↑ (more potassium) | faster repolarization | cleaner phrasing |
| `g_K` ↓ | sluggish repolarization | smeared, "wet" sound |

These are not just heuristics: each row can be derived from a perturbation analysis of the HH equations followed by the deterministic downstream chain. **The PINN is therefore an interpretable, biophysically grounded controller for the music.**

### C.4 Bidirectional learning

A future extension (Phase 8 stretch, mentioned in `ROADMAP.md`): the **inverse problem**. Given a target spectrogram `\hat S(f, t)` of the desired music, run gradient descent through the entire forward chain to recover the PINN parameters that produce the closest match. This is a standard inverse-problem use of PINNs (Raissi et al. 2019; NAML Lecture 27). It requires:

- Differentiable forward chain → use JAX everywhere, including a differentiable spike-event surrogate (e.g., temperature-relaxed thresholding).
- A differentiable spectrogram loss (STFT magnitude L² distance).
- A regularizer that keeps `(m_∞, τ_m)` biophysically plausible (e.g., Sobolev seminorm on the PINN's output functions, per NAML Lecture 25).

---

## Mathematical correspondences (summary table)

| Property | Worm side (Section A) | Piano side (Section B) |
|---|---|---|
| Dependent variable(s) | `V(t)`, gating `m,h,n` | `u(x,t)` string, `w(\mathbf{r},t)` plate |
| Spatial dimension | 0-D (point neuron) → 1-D (cable) → 3-D (body SPH) | 1-D (string) → 2-D (plate) → 3-D (air) |
| Time-order | 1 (gating), 2 (cable wave) | 2 (string, plate, air) |
| Space-order | 2 (cable) | 2 (string tension), 4 (string bending; plate biharmonic) |
| Linearity | nonlinear in gating | linear PDE + nonlinear hammer contact |
| Dispersion | spatial (in cable model) | inharmonicity from bending stiffness |
| Excitation | injected current `I_ext` | hammer force `F_h(t)\delta(x-x_h)` |
| Threshold | voltage crossing `V_θ` | felt compression `δ > 0` |
| Dominant timescales | `τ_m, τ_h, τ_n, τ_{syn}` (0.1–10 ms) | `1/(2σ_0)` (50 ms – 5 s decay) |
| Numerical method (wormuse) | NEURON (C302) + SPH (Sibernetic) + PINN (PyANNOW) | FDM 1D (Phase 2) → deal.II FEM (Phase 5) |
| Course alignment | NAML L14, L17, L27 + AppStat L01–L07 | AMSC L05–L12, NMPDE L1–L8, NLA full |

---

## References

### Biology
- Hodgkin, A. L., & Huxley, A. F. (1952). *A quantitative description of membrane current and its application to conduction and excitation in nerve.* The Journal of Physiology, 117(4), 500-544.
- White, J. G., Southgate, E., Thomson, J. N., & Brenner, S. (1986). *The structure of the nervous system of the nematode Caenorhabditis elegans.* Philosophical Transactions of the Royal Society B, 314(1165), 1-340.
- Cook, S. J., et al. (2019). *Whole-animal connectomes of both Caenorhabditis elegans sexes.* Nature, 571(7763), 63-71.
- Hill, A. V. (1938). *The heat of shortening and the dynamic constants of muscle.* Proceedings of the Royal Society B, 126(843), 136-195.
- Palyanov, A., Khayrulin, S., Larson, S. D., & Dibert, A. (2018). *Sibernetic: a software complex based on the Predictive-Corrective Incompressible SPH method for simulation of liquids and soft tissues.* PeerJ Computer Science, 4, e147.

### Piano physics
- Chabassier, J., Chaigne, A., & Joly, P. (2014). *Time domain simulation of a piano. Part 1: Model description.* ESAIM: M2AN, 48(5), 1241-1278. (Available in `polimuse-docs/`.)
- Chabassier, J., & Joly, P. (2014). *Time domain simulation of a piano. Part 2: Numerical aspects.* ESAIM: M2AN, 48(5), 1279-1313. (Available in `polimuse-docs/`.)
- Bensa, J., Bilbao, S., Kronland-Martinet, R., & Smith, J. O. (2003). *The simulation of piano string vibration: from physical models to finite difference schemes and digital waveguides.* JASA, 114(2), 1095-1107.

### Physics-Informed Neural Networks
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686-707.
- Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). *Physics-informed machine learning.* Nature Reviews Physics, 3(6), 422-440.

### Author's prior work (conceptual reference only — not vendored)
- *ChannelWorm-OW research proposal* — AppStat 2026 project, see `AppStat/project/VahidGhayoomie-channelworm-OW.pdf`.
- *SC-PINN: Structure-Conditioned PINN for Ion Channel Kinetics* — NAML 2025 project, see `Numerical_analysis_forML/naml-ion-channel-pinn/`.

---

## Document status

This is the scientific foundation for the wormuse project. It is intended to be **stable across phases** — implementation files in the sub-projects will reference equation numbers from this document (e.g., "implements equation A.2 with the θ-method from B.3"). When the equations evolve, update this document first and propagate references afterward.
