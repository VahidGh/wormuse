"""Step 8 — Physics-Informed Neural Network (locomotion mechanics).

NAML material:  Lecture 27  (PINNs — physics-informed neural networks)
                Lecture 14  (Autodiff — jax.grad, used for PDE residual)
                Lecture 22  (L-BFGS — standard two-stage PINN training)
                Lab 04      (JAX introduction)

The physics we encode: the worm's body is a **damped harmonic oscillator**.
Each muscle segment q_j(t) satisfies:

    q̈_j + 2γ q̇_j + ω² q_j = F_j(t)

where:
  - γ  = damping coefficient (energy dissipation in muscle)
  - ω  = natural frequency (= 2π × locomotion_freq_Hz)
  - F_j = neural force from the motor neurons (the composer's input)

The PINN composer predicts q_j(t) from the neural latent code z(t).
The physics loss penalises any prediction that violates the ODE above.

Total loss = data_loss (match Chopin) + λ_phys × physics_loss (obey ODE)

NAML L27 connection:
    This is exactly the PINN recipe: standard NN + physics residual at
    collocation points, computed via jax.grad.
    Compare to the user's SC-PINN: instead of HH kinetics dV/dt = f(V, m, h, n),
    here the ODE is the simpler second-order oscillator above.

The simplification is intentional:
    The HH model captures ion-channel detail irrelevant to music.
    The oscillator captures what matters musically: the rhythm is driven by
    a periodic forcing with frequency ω and is damped with rate γ.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax


class PINNComposer(nn.Module):
    """PINN-enhanced composer: predicts (note sequence, ODE solution).

    Architecture follows NAML L27 Fig. 1:
      - Input: (t, z)  where t = time and z = neural latent code
      - Output: q(t)   the predicted muscle activation (→ note sequence)
    """
    hidden:  int = 48
    depth:   int = 3
    out_dim: int = 8   # muscle groups

    @nn.compact
    def __call__(self, tz: jnp.ndarray) -> jnp.ndarray:
        """tz : (batch, 1 + k_worm)  →  q : (batch, out_dim)."""
        x = tz
        for _ in range(self.depth):
            x = nn.Dense(self.hidden)(x)
            x = nn.tanh(x)            # tanh is smoother for ODE derivatives
        return nn.Dense(self.out_dim)(x)


# ── Physics residual via jax.grad (L14 / L27) ────────────────────────────────

def osc_residual(params: dict, model: PINNComposer,
                  t: jnp.ndarray, z: jnp.ndarray,
                  gamma: float = 0.3, omega: float = 2.5) -> jnp.ndarray:
    """Compute the damped-oscillator residual at collocation points.

    Residual: q̈ + 2γ q̇ + ω² q - F  ≈ 0

    where  F = tanh(z @ W)  is a learned forcing function (the motor drive).

    Uses jax.grad twice to get q̈ from the network output — exactly the
    autodiff approach taught in L14.
    """
    def q_fn(t_scalar):
        tz = jnp.concatenate([t_scalar[None], z])
        return model.apply(params, tz[None]).squeeze(0)    # (out_dim,)

    # First derivative via jax.grad  (L14: reverse mode)
    dq_dt  = jax.jacfwd(q_fn)(t)                          # (out_dim,)
    # Second derivative
    d2q_dt = jax.jacfwd(jax.jacfwd(q_fn))(t)              # (out_dim,)

    tz     = jnp.concatenate([t[None], z])
    q      = model.apply(params, tz[None]).squeeze(0)

    # Forcing: linear function of neural latent (simple learnable coupling)
    F      = jnp.dot(z, jnp.ones(z.shape) / z.shape[0]) * jnp.ones(q.shape)

    residual = d2q_dt + 2.0 * gamma * dq_dt + omega**2 * q - F
    return residual


def pinn_loss(params: dict, model: PINNComposer,
               tz_data:      jnp.ndarray,  # (N_d, 1+k)
               q_data:       jnp.ndarray,  # (N_d, out_dim)
               t_colloc:     jnp.ndarray,  # (N_c,)  collocation times
               z_colloc:     jnp.ndarray,  # (N_c, k)  neural codes at colloc points
               lam_phys:     float = 0.1) -> jnp.ndarray:
    """Total PINN loss = data_loss + λ_phys × physics_loss.

    λ_phys is the key hyperparameter (trade-off between fitting Chopin and
    obeying the worm's mechanical law).  L27 suggests starting with λ=0.1.
    """
    # Data loss — match Chopin features at observed timesteps
    q_pred = model.apply(params, tz_data)
    data_loss = jnp.mean((q_pred - q_data)**2)

    # Physics loss — evaluate ODE residual at random collocation points
    res_fn = lambda i: osc_residual(params, model, t_colloc[i], z_colloc[i])
    # vmap over collocation points
    residuals  = jax.vmap(lambda t_, z_: osc_residual(params, model, t_, z_))(
                     t_colloc, z_colloc)
    phys_loss  = jnp.mean(residuals**2)

    return data_loss + lam_phys * phys_loss, (data_loss, phys_loss)


def train_pinn_adam_lbfgs(
    model:      PINNComposer,
    params:     dict,
    tz_data:    np.ndarray,
    q_data:     np.ndarray,
    t_colloc:   np.ndarray,
    z_colloc:   np.ndarray,
    lam_phys:   float = 0.1,
    adam_steps: int   = 1000,
    lbfgs_steps: int  = 100,
    lr:         float = 1e-3,
    verbose:    bool  = True,
) -> tuple[dict, dict]:
    """Standard PINN two-stage training: Adam warmup → L-BFGS polish.

    Identical recipe to L27 slides and the user's SC-PINN project.
    """
    tz_j = jnp.array(tz_data, dtype=jnp.float32)
    q_j  = jnp.array(q_data,  dtype=jnp.float32)
    tc_j = jnp.array(t_colloc, dtype=jnp.float32)
    zc_j = jnp.array(z_colloc, dtype=jnp.float32)

    loss_fn = lambda p: pinn_loss(p, model, tz_j, q_j, tc_j, zc_j, lam_phys)[0]

    # Stage 1: Adam
    opt = optax.adam(lr)
    state = opt.init(params)
    history_adam = []

    @jax.jit
    def adam_step(p, s):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, s  = opt.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss

    for step in range(adam_steps):
        params, state, loss = adam_step(params, state)
        history_adam.append(float(loss))
        if verbose and step % 100 == 0:
            print(f"  Adam {step:4d}  loss={float(loss):.5f}")

    # Stage 2: L-BFGS
    lbfgs = optax.lbfgs()
    lb_state = lbfgs.init(params)
    history_lbfgs = []

    for step in range(lbfgs_steps):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, lb_state = lbfgs.update(grads, lb_state, params,
                                          value=loss, grad=grads, value_fn=loss_fn)
        params = optax.apply_updates(params, updates)
        history_lbfgs.append(float(loss))
        if verbose and step % 20 == 0:
            print(f"  L-BFGS {step:3d}  loss={float(loss):.6f}")

    return params, {"adam": history_adam, "lbfgs": history_lbfgs}
