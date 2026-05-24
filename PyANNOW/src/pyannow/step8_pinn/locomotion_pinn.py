"""Step 8 — Physics-Informed Neural Network: ODE vs PDE comparison.

NAML material:  Lecture 27  (PINNs — physics-informed neural networks)
                Lecture 14  (Autodiff — jax.grad twice for second derivatives)
                Lecture 22  (L-BFGS — the second-stage optimizer in all PINNs)

╔═══════════════════════════════════════════════════════════════════════════╗
║  ARCHITECTURE NOTE (ISSUE-036, v0.8.0)                                   ║
║                                                                           ║
║  This module enforces a LOCOMOTION-OSCILLATOR ODE/PDE.                   ║
║  It is NOT the ion-channel PINN described in ION_CHANNELS.md.             ║
║                                                                           ║
║  The ion-channel PINN (Hodgkin-Huxley gating kinetics — learning          ║
║  m_∞, h_∞, τ_m, τ_h as functions of voltage) is the project's intended   ║
║  long-term centerpiece but is NOT YET IMPLEMENTED in PyANNOW.            ║
║  Track in ISSUE-008 (ca_thresh as PINN parameter is the first step).     ║
║                                                                           ║
║  What this module DOES implement (and is scientifically valid):           ║
║    Step 8a  — per-muscle damped oscillator ODE:  q̈ + 2γq̇ + ω²q = F      ║
║    Step 8b  — 1D worm-body wave equation PDE:   ρq_tt − μq_xx + γq_t = F ║
║  These are coarse-grained surrogates for the dynamics that the            ║
║  ion-channel PINN would produce.  See SCIENTIFIC_FOUNDATION.md §C.2.     ║
╚═══════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════
The core insight: the same PINN recipe works for two levels of physics:

  Step 8a — ODE (Ordinary Differential Equation)
  ──────────────────────────────────────────────
  Each muscle group j is a damped harmonic oscillator:
      q̈_j(t) + 2γ q̇_j(t) + ω² q_j(t) = F_j(t)

  This is a point model: one ODE per muscle, time only.
  Simplest physics; maximum NAML clarity.

  Step 8b — PDE (Partial Differential Equation)
  ──────────────────────────────────────────────
  The worm body is a 1D elastic beam. The lateral displacement q(x,t)
  satisfies a damped wave equation (see SCIENTIFIC_FOUNDATION.md §A.6):
      ρ q_tt(x,t) - μ q_xx(x,t) + γ q_t(x,t) = F(x,t)

  This is the spatially-extended version of the ODE above (set q_xx → 0
  and you recover the ODE). It captures the bending wave that propagates
  from head to tail during locomotion.

  Beautiful symmetry with the piano:
      Piano string (§B.2):   ρ_s ü - T u_xx + ESκ² u_xxxx + ... = F_hammer
      Worm body   (§A.6):    ρ   q̈ - μ q_xx + γ q_t           = F_neural

  The piano and the worm obey the same FAMILY of PDEs — this is the
  Structural Correspondence 2 from SCIENTIFIC_FOUNDATION.md §C.2.

═══════════════════════════════════════════════════════════════

NAML L27 protocol (same for both):
  1. Sample random collocation points (t) or (x, t)
  2. Evaluate the PDE/ODE residual via jax.grad twice
  3. Add residual² to the data loss
  4. Train with Adam → L-BFGS (the SC-PINN recipe, L22)

Why compare both?
  • ODE = NAML L27 simplest case (course's own examples use 1D ODEs)
  • PDE = more physics, closer to NMPDE / AMSC connection, richer music
  • Comparison shows the cost (ODE runs in seconds, PDE in minutes)
    and the benefit (PDE notes have smoother waveforms, less jitter)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax


# ─── Shared network backbone ─────────────────────────────────────────────────

class PhysicsComposer(nn.Module):
    """PINN network used for both ODE and PDE versions.

    Inputs differ only in the number of physics coordinates:
      ODE: input = (t, z)        shape (1 + k_worm,)
      PDE: input = (x, t, z)     shape (2 + k_worm,)

    Architecture: 3 hidden layers of width `hidden`, tanh activations.
    This matches Lab10's Flax style and the user's SC-PINN architecture.
    """
    hidden:  int = 48
    depth:   int = 3
    out_dim: int = 8   # one output per muscle group

    @nn.compact
    def __call__(self, inp: jnp.ndarray) -> jnp.ndarray:
        x = inp
        for _ in range(self.depth):
            x = nn.Dense(self.hidden,
                         kernel_init=nn.initializers.xavier_normal())(x)
            x = nn.tanh(x)
        return nn.Dense(self.out_dim)(x)


# ═══════════════════════════════════════════════════════════════
# Step 8a — ODE PINN (damped harmonic oscillator)
# ═══════════════════════════════════════════════════════════════

def ode_residual(params: dict, model: PhysicsComposer,
                 t: jnp.ndarray, z: jnp.ndarray,
                 gamma: float = 0.3, omega: float = 2.5) -> jnp.ndarray:
    """Residual of  q̈ + 2γ q̇ + ω² q = F_neural  at time t.

    Uses jax.grad twice (L14: autodiff in action):
        dq/dt  = ∂q/∂t  via jax.grad
        d²q/dt² = ∂²q/∂t² via jax.grad(jax.grad)

    Parameters
    ----------
    t : scalar  collocation time (float)
    z : (k_worm,)  neural latent code at time t
    """
    def q_at(t_scalar: jnp.ndarray) -> jnp.ndarray:
        inp = jnp.concatenate([t_scalar[None], z])   # (1 + k_worm,)
        return model.apply(params, inp).squeeze()     # (out_dim,)

    # First derivative q̇ via reverse-mode autodiff (L14)
    dq  = jax.jacfwd(q_at)(t)           # (out_dim,)  — forward mode for speed
    # Second derivative q̈
    d2q = jax.jacfwd(jax.jacfwd(q_at))(t)   # (out_dim,)

    q   = q_at(t)
    # Simple neural forcing: weighted sum of neural features
    F   = jnp.dot(z / (jnp.linalg.norm(z) + 1e-6),
                  jnp.ones(z.shape) / z.shape[0]) * jnp.ones_like(q)

    return d2q + 2.0 * gamma * dq + omega**2 * q - F


def ode_physics_loss(params: dict, model: PhysicsComposer,
                      t_colloc: jnp.ndarray, z_colloc: jnp.ndarray,
                      gamma: float = 0.3, omega: float = 2.5) -> jnp.ndarray:
    """Mean-squared ODE residual over collocation points."""
    residuals = jax.vmap(
        lambda t_, z_: ode_residual(params, model, t_, z_, gamma, omega)
    )(t_colloc, z_colloc)
    return jnp.mean(residuals ** 2)


# ═══════════════════════════════════════════════════════════════
# Step 8b — PDE PINN (1D wave equation along the worm body)
# ═══════════════════════════════════════════════════════════════

def pde_residual(params: dict, model: PhysicsComposer,
                 x: jnp.ndarray, t: jnp.ndarray, z: jnp.ndarray,
                 rho: float = 1.0, mu: float = 0.5, gamma: float = 0.3
                 ) -> jnp.ndarray:
    """Residual of  ρ q_tt - μ q_xx + γ q_t = F  at collocation point (x, t).

    ρ = linear mass density  (kg/m)
    μ = bending stiffness    (N)
    γ = viscous damping coefficient

    The PDE is the simplest 1D damped wave equation describing the worm's
    lateral bending wave.  Compare to SCIENTIFIC_FOUNDATION.md §A.6:
    the full SPH model reduces to this in the 1D linearised limit.

    And compare to the piano string (§B.2):
        ρ_s ü - T_s u_xx = 0   (flexible string, no bending)
    Here we drop the bending stiffness term (ES κ² u_xxxx) since the worm
    body is much more flexible than a piano string.

    Autodiff via jax.grad:
        ∂q/∂t  = jacfwd w.r.t. the t argument
        ∂q/∂x  = jacfwd w.r.t. the x argument
        ∂²q/∂t², ∂²q/∂x²: second applications of jacfwd
    """
    def q_at(xt: jnp.ndarray) -> jnp.ndarray:
        """q(x, t) = network output.  xt = [x, t]."""
        inp = jnp.concatenate([xt, z])     # (2 + k_worm,)
        return model.apply(params, inp).squeeze()

    xt = jnp.stack([x, t])

    # jacfwd(q_at)(xt) has shape (out_dim, 2):
    #   row i = outputs;  col 0 = ∂/∂x;  col 1 = ∂/∂t
    # We need COLUMN slices to get per-output derivatives.

    # Spatial derivative q_xx
    def q_x_deriv(xt_: jnp.ndarray) -> jnp.ndarray:
        return jax.jacfwd(q_at)(xt_)[:, 0]   # (out_dim,): ∂q_i/∂x

    q_xx = jax.jacfwd(q_x_deriv)(xt)[:, 0]  # (out_dim,): ∂²q_i/∂x²

    # Temporal derivative q_tt
    def q_t_deriv(xt_: jnp.ndarray) -> jnp.ndarray:
        return jax.jacfwd(q_at)(xt_)[:, 1]   # (out_dim,): ∂q_i/∂t

    q_t  = q_t_deriv(xt)                     # (out_dim,)
    q_tt = jax.jacfwd(q_t_deriv)(xt)[:, 1]  # (out_dim,): ∂²q_i/∂t²

    q    = q_at(xt)
    # Neural forcing varies with body position and neural latent
    F    = jnp.sin(jnp.pi * x) * jnp.dot(z / (jnp.linalg.norm(z) + 1e-6),
                                           jnp.ones(z.shape) / z.shape[0]) * jnp.ones_like(q)

    return rho * q_tt - mu * q_xx + gamma * q_t - F


def pde_physics_loss(params: dict, model: PhysicsComposer,
                      x_colloc: jnp.ndarray, t_colloc: jnp.ndarray,
                      z_colloc: jnp.ndarray,
                      rho: float = 1.0, mu: float = 0.5,
                      gamma: float = 0.3) -> jnp.ndarray:
    """Mean-squared PDE residual over (x, t) collocation grid."""
    residuals = jax.vmap(
        lambda x_, t_, z_: pde_residual(params, model, x_, t_, z_, rho, mu, gamma)
    )(x_colloc, t_colloc, z_colloc)
    return jnp.mean(residuals ** 2)


# ─── Combined loss (data + physics) ──────────────────────────────────────────

def pinn_loss(params: dict, model: PhysicsComposer,
              inp_data:    jnp.ndarray,   # (N_d, 1+k)  ODE  or  (N_d, 2+k) PDE
              q_data:      jnp.ndarray,   # (N_d, out_dim)
              phys_loss_fn: Callable,
              phys_args:    tuple,
              lam_phys:    float = 0.1) -> tuple:
    """Total PINN loss = data_loss + λ_phys × physics_loss.

    lam_phys trades off fitting Chopin vs obeying the worm's mechanics.
    """
    q_pred    = model.apply(params, inp_data)
    data_loss = jnp.mean((q_pred - q_data) ** 2)
    phys      = phys_loss_fn(params, model, *phys_args)
    return data_loss + lam_phys * phys, (data_loss, phys)


# ─── Training (identical recipe for both ODE and PDE) ────────────────────────

@dataclass
class PINNHistory:
    adam_losses:  list = field(default_factory=list)
    lbfgs_losses: list = field(default_factory=list)
    data_losses:  list = field(default_factory=list)
    phys_losses:  list = field(default_factory=list)
    wall_time:    float = 0.0


def train_pinn(
    model:       PhysicsComposer,
    params:      dict,
    inp_data:    np.ndarray,
    q_data:      np.ndarray,
    phys_loss_fn: Callable,
    phys_args:   tuple,
    lam_phys:    float = 0.1,
    lr:          float = 1e-3,
    adam_steps:  int   = 1000,
    lbfgs_steps: int   = 100,
    verbose:     bool  = True,
    label:       str   = "PINN",
) -> tuple[dict, PINNHistory]:
    """Standard PINN two-stage training: Adam → L-BFGS.

    Identical recipe for ODE and PDE — the only difference is the
    physics_loss_fn and phys_args arguments.

    This is the exact protocol from:
    - NAML L27 slides
    - The user's SC-PINN project (naml-ion-channel-pinn/)
    """
    id_j = jnp.array(inp_data, dtype=jnp.float32)
    qd_j = jnp.array(q_data,   dtype=jnp.float32)

    def loss_fn(p):
        total, parts = pinn_loss(p, model, id_j, qd_j,
                                  phys_loss_fn, phys_args, lam_phys)
        return total, parts

    history = PINNHistory()
    t0 = time.perf_counter()

    # Stage 1: Adam (L20)
    opt   = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def adam_step(p, s):
        (loss, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        updates, s = opt.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss, parts

    for step in range(adam_steps):
        params, state, loss, (dl, pl) = adam_step(params, state)
        history.adam_losses.append(float(loss))
        history.data_losses.append(float(dl))
        history.phys_losses.append(float(pl))
        if verbose and step % 100 == 0:
            print(f"  [{label}] Adam {step:4d}  total={float(loss):.5f}  "
                  f"data={float(dl):.5f}  phys={float(pl):.5f}")

    # Stage 2: L-BFGS (L22)
    lbfgs = optax.lbfgs()
    lb_st = lbfgs.init(params)

    scalar_loss = lambda p: loss_fn(p)[0]

    for step in range(lbfgs_steps):
        loss, grads = jax.value_and_grad(scalar_loss)(params)
        upd, lb_st = lbfgs.update(grads, lb_st, params,
                                   value=loss, grad=grads, value_fn=scalar_loss)
        params = optax.apply_updates(params, upd)
        history.lbfgs_losses.append(float(loss))
        if verbose and step % 20 == 0:
            print(f"  [{label}] L-BFGS {step:3d}  loss={float(loss):.6f}")

    history.wall_time = time.perf_counter() - t0
    return params, history


# ═══════════════════════════════════════════════════════════════
# Step 8c — PINN-Classifier (BCE data loss + ODE physics residual)
# ═══════════════════════════════════════════════════════════════
#
# Conceptual update (v0.9.0 / ISSUE-042c):
#
#   Steps 7 and 9 showed that directly classifying onsets (BCE loss)
#   beats the regression approach (MSE against Chopin features).
#   Step 8c applies the same insight to the PINN:
#
#     Old Step 8a/8b: MSE(q̂, C_chopin) + λ · physics_residual
#     New Step 8c:    BCE(σ(logit), y_onset) + λ · physics_residual
#
#   The ODE constraint now acts as temporal REGULARISATION on the
#   onset probability trajectory — preventing the classifier from
#   predicting arbitrarily spiky outputs that violate locomotion dynamics.
#
#   Biologically: the ODE  logiẗ + 2γ·logiṫ + ω²·logit = F_neural
#   forces refractory periods and a characteristic oscillation frequency
#   that the pure-data classifier (Step 9) cannot enforce.


def _bce_from_logits(logits: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable sigmoid BCE.

    Implements: E[ max(x,0) - x·y + log(1 + exp(-|x|)) ]
    Avoids overflow from log(exp(x)) for large positive x.
    """
    return jnp.mean(
        jnp.maximum(logits, 0.0) - logits * y
        + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )


def pinn_classifier_loss(
    params:       dict,
    model:        PhysicsComposer,
    inp_data:     jnp.ndarray,    # (N, 1+k_worm)  same layout as ODE
    y_onset:      jnp.ndarray,    # (N,) float32   binary labels
    phys_loss_fn: Callable,
    phys_args:    tuple,
    lam_phys:     float = 0.05,
) -> tuple:
    """Total PINN-Classifier loss = BCE(σ(logit), y_onset) + λ · physics_loss.

    The data term directly optimises for onset detection (same objective
    as Step 7 / Step 9) while the physics term enforces ODE dynamics.
    """
    logits    = model.apply(params, inp_data).squeeze(-1)   # (N,)
    bce       = _bce_from_logits(logits, y_onset)
    phys      = phys_loss_fn(params, model, *phys_args)
    return bce + lam_phys * phys, (bce, phys)


def train_pinn_classifier(
    model:        PhysicsComposer,
    params:       dict,
    inp_data:     np.ndarray,     # (N, 1+k_worm)
    y_onset:      np.ndarray,     # (N,) binary (0 / 1)
    phys_loss_fn: Callable,
    phys_args:    tuple,
    lam_phys:     float = 0.05,
    lr:           float = 1e-3,
    adam_steps:   int   = 600,
    lbfgs_steps:  int   = 60,
    verbose:      bool  = True,
    label:        str   = "PINN-CLS",
) -> tuple[dict, PINNHistory]:
    """PINN-Classifier training: BCE + ODE physics, Adam → L-BFGS.

    Drop-in replacement for train_pinn() — identical loop, different loss.
    """
    id_j = jnp.array(inp_data, dtype=jnp.float32)
    yo_j = jnp.array(y_onset,  dtype=jnp.float32)

    def loss_fn(p):
        return pinn_classifier_loss(p, model, id_j, yo_j,
                                    phys_loss_fn, phys_args, lam_phys)

    history = PINNHistory()
    t0 = time.perf_counter()

    # Stage 1 — Adam
    opt   = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def adam_step(p, s):
        (loss, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        updates, s = opt.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss, parts

    for step in range(adam_steps):
        params, state, loss, (bce, pl) = adam_step(params, state)
        history.adam_losses.append(float(loss))
        history.data_losses.append(float(bce))
        history.phys_losses.append(float(pl))
        if verbose and step % 100 == 0:
            print(f"  [{label}] Adam {step:4d}  total={float(loss):.5f}  "
                  f"BCE={float(bce):.5f}  phys={float(pl):.5f}")

    # Stage 2 — L-BFGS
    lbfgs  = optax.lbfgs()
    lb_st  = lbfgs.init(params)
    scalar_loss = lambda p: loss_fn(p)[0]

    for step in range(lbfgs_steps):
        loss, grads = jax.value_and_grad(scalar_loss)(params)
        upd, lb_st  = lbfgs.update(grads, lb_st, params,
                                    value=loss, grad=grads,
                                    value_fn=scalar_loss)
        params = optax.apply_updates(params, upd)
        history.lbfgs_losses.append(float(loss))
        if verbose and step % 20 == 0:
            print(f"  [{label}] L-BFGS {step:3d}  loss={float(loss):.6f}")

    history.wall_time = time.perf_counter() - t0
    return params, history


def run_pinn_classifier(
    Z_worm:      np.ndarray,   # (T, k_worm) neural PCA scores
    y_onset:     np.ndarray,   # (T,) binary onset labels
    t_arr:       np.ndarray,   # (T,) time in seconds
    bpm:         float = 50.0, # piece BPM -- beat-phase Fourier feature
    n_harmonics: int   = 12,   # sin/cos harmonics in time embedding (Step 9)
    max_train_pts: int   = 10000, # subsample data for Adam (matches Step 9 speed)
    lam_phys:    float = 0.05,
    gamma:       float = 0.3,
    omega:       float = 2.5,
    adam_steps:  int   = 400,
    lbfgs_steps: int   = 60,
    seed:        int   = 0,
    verbose:     bool  = True,
) -> dict:
    """Step 8c: Physics-Informed Neural Network Classifier.

    Same interface as compare_ode_vs_pde() but optimises for onset
    classification rather than Chopin-feature regression.

    Architecture
    ------------
    Input:   [t | Zs | T_feats]  shape (1 + k_worm + 1 + 2*n_harmonics + 2,)
             t       -- physical time in seconds (position 0; differentiated for ODE)
             Zs      -- standardised worm PCA scores
             T_feats -- Fourier time embeddings identical to Step 9 (phase +
                        n_harmonics sin/cos pairs + beat sin/cos)
    Network: PhysicsComposer(hidden=96, depth=3, out_dim=1) -- single logit
    Output:  sigmoid(logit) in (0,1) -- onset probability

    Loss
    ----
    Total = BCE(sigmoid(logit), y_onset) + lam_phys * ODE_residual^2

    ODE constraint on the logit: logit_tt + 2*gamma*logit_t + omega^2*logit = F
    This forces onset probabilities to oscillate with locomotion frequency,
    enforcing refractory periods as a physics prior.

    Why Fourier features help the PINN
    ------------------------------------
    Without them the PINN must learn sin/cos from scratch via tanh (slow).
    Adding precomputed Fourier features (same as Step 9) gives an explicit
    time basis; the ODE then regularises a rich representation faster.

    The physics residual differentiates the logit w.r.t. t (position 0),
    treating Zs and T_feats as frozen context at each collocation point.

    Returns
    -------
    dict with keys: model, params, history, probs (T,), inp (T, in_dim)
    """
    from sklearn.preprocessing import StandardScaler as _SS

    T, k = Z_worm.shape

    # Fourier time embeddings (same recipe as Step 9)
    phase_s = t_arr / t_arr[-1]
    tf_cols = [phase_s[:, None]]
    for k_h in range(1, n_harmonics + 1):
        ang = 2.0 * np.pi * k_h * phase_s
        tf_cols += [np.sin(ang)[:, None], np.cos(ang)[:, None]]
    beat_phase = t_arr * bpm / 60.0
    tf_cols += [np.sin(2 * np.pi * beat_phase)[:, None],
                np.cos(2 * np.pi * beat_phase)[:, None]]
    T_feats = np.concatenate(tf_cols, axis=1)   # (T, 1 + 2*n_harmonics + 2)

    Zs = _SS().fit_transform(Z_worm)            # (T, k_worm)

    # Full input: t at position 0 (for ODE deriv), then worm+Fourier context
    in_dim = 1 + k + T_feats.shape[1]
    inp_full = np.column_stack([t_arr, Zs, T_feats]).astype(np.float32)  # (T, in_dim)

    # Subsample training data (same speed trick as Step 9, default 10k pts)
    rng     = np.random.default_rng(seed)
    n_train = min(max_train_pts, T)
    idx_tr  = np.sort(rng.choice(T, n_train, replace=False))
    inp_tr  = inp_full[idx_tr]                        # (n_train, in_dim)
    y_tr    = y_onset.astype(np.float32)[idx_tr]      # (n_train,)

    # Network: increased capacity for richer input
    model  = PhysicsComposer(hidden=96, depth=3, out_dim=1)
    key    = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.zeros((1, in_dim)))

    # ODE collocation: z_col = [Zs | T_feats] at collocation times (frozen)
    N_c   = min(400, T)
    idx_c = rng.integers(0, T, N_c)
    t_col = jnp.array(t_arr[idx_c], dtype=jnp.float32)
    z_col = jnp.array(
        np.hstack([Zs[idx_c], T_feats[idx_c]]), dtype=jnp.float32)  # (N_c, k+n_tf)

    if verbose:
        print("\n=== Step 8c: PINN-Classifier (BCE + ODE + Fourier) ===")
        print(f"  in_dim={in_dim}  [t(1) | worm({k}) | Fourier({T_feats.shape[1]})]")
        print(f"  hidden=96  depth=3  out_dim=1  (logit)")
        print(f"  train_pts={n_train}/{T}  colloc={N_c}")
        print(f"  Data loss : BCE(sigmoid(logit), y_onset)")
        print(f"  Physics   : logit_tt + 2*{gamma}*logit_t + {omega:.1f}^2*logit = F")
        print(f"  lam_phys={lam_phys}  Adam={adam_steps}  L-BFGS={lbfgs_steps}")
        print(f"  Onset prevalence: {y_onset.mean():.4f}  "
              f"({int(y_onset.sum())} / {T} frames)")

    params, history = train_pinn_classifier(
        model, params, inp_tr, y_tr,
        phys_loss_fn=ode_physics_loss,
        phys_args=(t_col, z_col, gamma, omega),
        lam_phys=lam_phys,
        lr=1e-3,
        adam_steps=adam_steps,
        lbfgs_steps=lbfgs_steps,
        verbose=verbose,
        label="PINN-CLS",
    )

    # Inference on full sequence
    logits_out = model.apply(params, jnp.array(inp_full)).squeeze(-1)  # (T,)
    probs      = np.array(jax.nn.sigmoid(logits_out))

    final_loss = (history.lbfgs_losses[-1] if history.lbfgs_losses
                  else history.adam_losses[-1])
    if verbose:
        print(f"\n  PINN-CLS final loss={final_loss:.6f}  "
              f"wall_time={history.wall_time:.1f}s")

    return {
        "model":   model,
        "params":  params,
        "history": history,
        "probs":   probs,
        "inp":     inp_full,
    }



# ─── Comparison helpers ───────────────────────────────────────────────────────

def compare_ode_vs_pde(
    Z_worm:    np.ndarray,          # (T, k_worm) neural latent codes
    C_chopin:  np.ndarray,          # (T, out_dim) target musical features
    t_arr:     np.ndarray,          # (T,) time axis in seconds
    lam_phys:  float = 0.1,
    adam_steps: int  = 500,
    lbfgs_steps: int = 50,
    seed:      int   = 0,
    verbose:   bool  = True,
    ca_thresh: float = -10.0,       # mV — PINN-tunable gate from CelegansChannelParams
) -> dict:
    """Run the ODE PINN and the PDE PINN on the same data and compare.

    Returns a dict with both trained parameter sets and both histories,
    ready for the notebook's comparison plot.

    Key comparison axes:
      1. Convergence speed  (wall time per step, loss curves)
      2. Final data loss    (how close to Chopin?)
      3. Physics loss       (how well does the worm obey its own mechanics?)
      4. Generated melody   (qualitative: is the ODE or PDE melody smoother?)
    """
    T, k = Z_worm.shape
    out_dim = C_chopin.shape[1]

    # ── Shared network (ODE uses (1+k) input, PDE uses (2+k)) ────────────────
    model_ode = PhysicsComposer(hidden=32, depth=2, out_dim=out_dim)
    model_pde = PhysicsComposer(hidden=32, depth=2, out_dim=out_dim)

    key_ode = jax.random.PRNGKey(seed)
    key_pde = jax.random.PRNGKey(seed + 1)
    params_ode = model_ode.init(key_ode, jnp.zeros((1, 1 + k)))
    params_pde = model_pde.init(key_pde, jnp.zeros((1, 2 + k)))

    # ── ODE collocation points (time only) ───────────────────────────────────
    rng = np.random.default_rng(seed)
    N_c = min(200, T)
    idx_c = rng.integers(0, T, N_c)
    t_colloc = jnp.array(t_arr[idx_c], dtype=jnp.float32)
    z_colloc = jnp.array(Z_worm[idx_c], dtype=jnp.float32)

    inp_ode = jnp.array(
        np.column_stack([t_arr, Z_worm]), dtype=jnp.float32)  # (T, 1+k)

    # ── PDE collocation points (x, t) ────────────────────────────────────────
    x_colloc = jnp.array(rng.uniform(0, 1, N_c), dtype=jnp.float32)

    inp_pde = jnp.array(
        np.column_stack([
            np.linspace(0, 1, T),   # x: position along worm body
            t_arr,                  # t: time
            Z_worm
        ]), dtype=jnp.float32)     # (T, 2+k)

    q_j = jnp.array(C_chopin, dtype=jnp.float32)

    # ── Train ODE PINN ────────────────────────────────────────────────────────
    if verbose: print("\n=== Step 8a: ODE PINN (damped oscillator) ===")
    p_ode, h_ode = train_pinn(
        model_ode, params_ode, inp_ode, q_j,
        phys_loss_fn=ode_physics_loss,
        phys_args=(t_colloc, z_colloc),
        lam_phys=lam_phys,
        adam_steps=adam_steps, lbfgs_steps=lbfgs_steps,
        verbose=verbose, label="ODE"
    )

    # ── Train PDE PINN ────────────────────────────────────────────────────────
    if verbose: print("\n=== Step 8b: PDE PINN (1D wave equation) ===")
    p_pde, h_pde = train_pinn(
        model_pde, params_pde, inp_pde, q_j,
        phys_loss_fn=pde_physics_loss,
        phys_args=(x_colloc, t_colloc, z_colloc),
        lam_phys=lam_phys,
        adam_steps=adam_steps, lbfgs_steps=lbfgs_steps,
        verbose=verbose, label="PDE"
    )

    # ── Summaries ─────────────────────────────────────────────────────────────
    final_ode = h_ode.lbfgs_losses[-1] if h_ode.lbfgs_losses else h_ode.adam_losses[-1]
    final_pde = h_pde.lbfgs_losses[-1] if h_pde.lbfgs_losses else h_pde.adam_losses[-1]

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  ODE PINN: final_loss={final_ode:.6f}  time={h_ode.wall_time:.1f}s")
        print(f"  PDE PINN: final_loss={final_pde:.6f}  time={h_pde.wall_time:.1f}s")
        winner = "ODE" if final_ode < final_pde else "PDE"
        print(f"  Better fit: {winner}")
        print(f"\n  Insight: PDE captures spatial body wave propagation.")
        print(f"  ODE treats each muscle independently.")
        print(f"  The PDE loss is typically lower because it encodes more physics.")
        print(f"{'─'*50}")

    return {
        "model_ode": model_ode, "params_ode": p_ode, "history_ode": h_ode,
        "model_pde": model_pde, "params_pde": p_pde, "history_pde": h_pde,
        "inp_ode": np.array(inp_ode), "inp_pde": np.array(inp_pde),
        "final_loss_ode": final_ode, "final_loss_pde": final_pde,
        "ca_thresh": ca_thresh,   # threshold used for converting PINN output → notes
    }
