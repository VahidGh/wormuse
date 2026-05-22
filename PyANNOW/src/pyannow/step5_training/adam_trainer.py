"""Step 5 — Adam / mini-batch SGD training pipeline.

NAML material:  Lecture 18  (Gradient Descent — step size, convergence)
                Lecture 19  (SGD — stochastic gradients, mini-batches)
                Lecture 20  (SGD variants: momentum, AdaGrad, RMSProp, Adam)
                Lab 05      (GD + SGD on regression tasks)
                Lab 07      (California housing — full training loop with Adam)
                Lab 08      (Part 1: 1st-order optimization comparison)

Training recipe follows the Lab10 canonical pipeline:
  1. Adam first-stage (L20): fast convergence to a good basin
  2. Learning-rate schedule: decay after plateau
  3. Early stopping on validation set
  4. Training curves (loss vs epoch) — essential for diagnosing issues

NAML insight (L18-L20):
    Plain GD converges linearly at rate (1 - μ/L).
    Adam adapts the learning rate per parameter, giving robustness to
    different neural feature scales — crucial when some neural PCA dimensions
    carry 80% of the variance and others carry <1%.

Adam pseudocode (exactly as in L20 slides):
    m ← β₁m + (1-β₁) g        (first moment)
    v ← β₂v + (1-β₂) g²       (second moment)
    m̂ = m / (1 - β₁ᵗ)         (bias correction)
    v̂ = v / (1 - β₂ᵗ)
    θ ← θ - α m̂ / (√v̂ + ε)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..step4_ffnn.jax_composer import WormComposer, mse_loss


@dataclass
class TrainingHistory:
    train_loss: list = field(default_factory=list)
    val_loss:   list = field(default_factory=list)
    epochs:     list = field(default_factory=list)
    wall_time:  list = field(default_factory=list)

    def update(self, epoch: int, tr: float, va: float, t: float):
        self.epochs.append(epoch)
        self.train_loss.append(tr)
        self.val_loss.append(va)
        self.wall_time.append(t)


def train_adam(
    model:    WormComposer,
    params:   dict,
    Z:        np.ndarray,
    C:        np.ndarray,
    lr:       float = 1e-3,
    epochs:   int   = 500,
    batch_sz: int   = 32,
    val_frac: float = 0.15,
    patience: int   = 40,
    seed:     int   = 0,
    verbose:  bool  = True,
) -> tuple[dict, TrainingHistory]:
    """Train the WormComposer with mini-batch Adam (Lab10 / Lab07 pattern).

    Parameters
    ----------
    model    : WormComposer (created in step4)
    params   : initial parameters from init_params()
    Z        : (T, k_worm)   worm neural latent codes
    C        : (T, k_chopin) Chopin musical target features
    lr       : Adam learning rate (default 1e-3, as in Lab10)
    epochs   : max number of full passes over the data
    batch_sz : mini-batch size (Lab10 uses 32 for MNIST)
    val_frac : fraction of data held out for validation
    patience : early stopping patience (in epochs)
    """
    # ── Train / validation split ──────────────────────────────────────────────
    rng   = np.random.default_rng(seed)
    n     = len(Z)
    n_val = max(1, int(n * val_frac))
    idx   = rng.permutation(n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    Z_tr, C_tr = jnp.array(Z[tr_idx], dtype=jnp.float32), jnp.array(C[tr_idx], dtype=jnp.float32)
    Z_va, C_va = jnp.array(Z[val_idx], dtype=jnp.float32), jnp.array(C[val_idx], dtype=jnp.float32)

    # ── Adam optimizer (optax — Lab10 canonical) ──────────────────────────────
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    # ── JIT-compiled step (autodiff — L14) ───────────────────────────────────
    @jax.jit
    def step_fn(p, s, zb, cb):
        loss, grads = jax.value_and_grad(mse_loss)(p, model, zb, cb)
        updates, s  = opt.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss

    # ── Training loop ─────────────────────────────────────────────────────────
    history     = TrainingHistory()
    best_val    = float("inf")
    best_params = params
    no_improve  = 0
    t_start     = time.perf_counter()
    n_tr        = len(Z_tr)

    for epoch in range(1, epochs + 1):
        # Shuffle mini-batches
        perm = jnp.array(rng.permutation(n_tr))
        tr_loss_ep = 0.0
        n_batches  = max(1, n_tr // batch_sz)

        for b in range(n_batches):
            bi       = perm[b * batch_sz: (b + 1) * batch_sz]
            params, opt_state, bl = step_fn(params, opt_state, Z_tr[bi], C_tr[bi])
            tr_loss_ep += float(bl)

        tr_loss_ep /= n_batches
        va_loss     = float(mse_loss(params, model, Z_va, C_va))

        history.update(epoch, tr_loss_ep, va_loss, time.perf_counter() - t_start)

        # Early stopping
        if va_loss < best_val - 1e-6:
            best_val    = va_loss
            best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}  val_loss={va_loss:.5f}")
                break

        if verbose and epoch % 50 == 0:
            print(f"  epoch {epoch:4d}  train={tr_loss_ep:.5f}  val={va_loss:.5f}")

    return best_params, history
