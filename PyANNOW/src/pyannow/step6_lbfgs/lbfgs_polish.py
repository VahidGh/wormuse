"""Step 6 — L-BFGS fine-tuning (second-stage polish).

NAML material:  Lecture 21  (Newton's method — second-order convergence)
                Lecture 22  (Quasi-Newton, BFGS, L-BFGS)
                Lab 08      (Part 2: Newton method, comparison with GD)
                The user's own SC-PINN project — uses the same Adam → L-BFGS pattern

Why L-BFGS after Adam?
    Adam is first-order (uses only gradients → linear convergence near optimum).
    L-BFGS approximates the Hessian using past gradient differences → super-linear
    convergence.  The two-stage recipe is the standard in the PINN literature
    (and in the user's naml-ion-channel-pinn project).

NAML L22 insight:
    BFGS update:  H_{k+1} = (I - ρₖ sₖ yₖᵀ) Hₖ (I - ρₖ yₖ sₖᵀ) + ρₖ sₖ sₖᵀ
    L-BFGS stores only the last m pairs (s, y) — memory O(mk) instead of O(k²).

Lab08 connection:
    In Lab08 Part 2 we compared Newton (exact Hessian) vs GD on a quadratic.
    Here L-BFGS gives the benefits of Newton without computing the full 302×302 Hessian.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..step4_ffnn.jax_composer import WormComposer, mse_loss
from ..step5_training.adam_trainer import TrainingHistory


def polish_lbfgs(
    model:        WormComposer,
    params:       dict,
    Z:            np.ndarray,
    C:            np.ndarray,
    max_steps:    int   = 200,
    memory_size:  int   = 10,
    tol:          float = 1e-7,
    verbose:      bool  = True,
) -> tuple[dict, TrainingHistory]:
    """L-BFGS fine-tuning on the full training set.

    Parameters
    ----------
    model       : trained WormComposer from Step 5
    params      : Adam-warmed parameters (starting point for L-BFGS)
    Z, C        : full training data (not mini-batches — L-BFGS is full-batch)
    max_steps   : maximum number of L-BFGS iterations
    memory_size : m = number of vector pairs to store (typically 10-20)
    tol         : stop if gradient norm < tol
    """
    Z_j = jnp.array(Z, dtype=jnp.float32)
    C_j = jnp.array(C, dtype=jnp.float32)

    lbfgs = optax.lbfgs(memory_size=memory_size)
    state = lbfgs.init(params)

    loss_fn = lambda p: mse_loss(p, model, Z_j, C_j)

    history   = TrainingHistory()
    t_start   = time.perf_counter()

    for step in range(max_steps):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = float(jnp.sqrt(sum(jnp.sum(g**2) for g in
                                        jax.tree_util.tree_leaves(grads))))

        updates, state = lbfgs.update(
            grads, state, params,
            value=loss,
            grad=grads,
            value_fn=loss_fn,
        )
        params = optax.apply_updates(params, updates)

        history.update(step, float(loss), float(loss), time.perf_counter() - t_start)

        if verbose and step % 20 == 0:
            print(f"  L-BFGS step {step:3d}  loss={float(loss):.6f}  |∇|={grad_norm:.4e}")

        if grad_norm < tol:
            if verbose:
                print(f"  Converged at step {step}  |∇|={grad_norm:.2e} < tol={tol}")
            break

    return params, history


def compare_optimizers_on_toy(k: int = 4, n: int = 200,
                               seed: int = 0) -> dict:
    """Demonstrate convergence speed of GD vs Adam vs L-BFGS on a small problem.

    Lab08 analogy: we compare methods on a synthetic quadratic — here on
    a tiny worm-to-music regression problem.
    """
    from ..step4_ffnn.jax_composer import create_model, init_params
    from ..step5_training.adam_trainer import train_adam

    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal((n, k)).astype(np.float32)
    W   = rng.standard_normal((k, k)).astype(np.float32)
    C   = (Z @ W + 0.1 * rng.standard_normal((n, k))).astype(np.float32)

    model  = create_model(k_worm=k, k_chopin=k, hidden=16, depth=1)
    params = init_params(model, k)

    # Adam stage
    p_adam, h_adam = train_adam(model, params, Z, C,
                                lr=1e-3, epochs=300, verbose=False)
    # L-BFGS stage
    p_lbfgs, h_lbfgs = polish_lbfgs(model, p_adam, Z, C,
                                      max_steps=50, verbose=False)

    return {
        "adam_final_loss":  h_adam.train_loss[-1] if h_adam.train_loss else None,
        "lbfgs_final_loss": h_lbfgs.train_loss[-1] if h_lbfgs.train_loss else None,
        "adam_history":     h_adam,
        "lbfgs_history":    h_lbfgs,
    }
