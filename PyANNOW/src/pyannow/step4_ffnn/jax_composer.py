"""Step 4 — Feed-Forward Neural Network composer (JAX / Flax).

NAML material:  Lecture 14 (Automatic differentiation — reverse mode)
                Lecture 15 (Activation functions: tanh, ReLU, Softplus)
                Lecture 16 (Neural networks — architecture, Xavier init)
                Lecture 17 (Training: forward pass, backprop = autodiff)
                Lab 06     (ANN for XOR — the simplest possible NN)
                Lab 10     (MNIST with full Flax/Optax pipeline)

We replace the linear ridge mapping with a non-linear MLP:

    f_θ : ℝ^{k_worm} → ℝ^{k_chopin}

Architecture (mirrors Lab10's Flax patterns):
    Linear(k_worm → hidden) → tanh → Linear(hidden → hidden) → tanh → Linear(hidden → k_chopin)

NAML insight:
    The Universal Approximation Theorem (L24) guarantees that a 1-hidden-layer
    MLP can represent ANY continuous function — including the worm-to-Chopin map.
    The theorem says it EXISTS; optimisation (Steps 5-6) finds it.

Lab06 analogy:
    Lab06 solved XOR with a 2-layer network: 2 inputs → 2 hidden → 1 output.
    Here we solve "worm state → musical feature" with the same architecture
    but larger (k_worm inputs → k_chopin outputs).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class WormComposer(nn.Module):
    """MLP that maps a worm neural latent vector to a musical feature vector.

    Designed to match Lab10's Flax coding style (compact, explicit layers).
    Uses tanh activations (better for bounded musical features than ReLU).

    Parameters
    ----------
    hidden   : hidden layer width (start at 32, increase if underfitting)
    depth    : number of hidden layers (1-3; deeper = harder to train)
    out_dim  : Chopin musical feature dimension k_chopin
    """
    hidden:  int = 32
    depth:   int = 2
    out_dim: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.  x : (batch, k_worm)  →  out : (batch, out_dim)."""
        # Xavier init (L16: zero-mean, variance = 2/(n_in + n_out))
        kernel_init = nn.initializers.xavier_normal()

        for _ in range(self.depth):
            x = nn.Dense(self.hidden, kernel_init=kernel_init)(x)
            x = nn.tanh(x)                    # L15: tanh avoids dying neurons, bounded output
        return nn.Dense(self.out_dim, kernel_init=kernel_init)(x)


def create_model(k_worm: int, k_chopin: int,
                  hidden: int = 32, depth: int = 2) -> WormComposer:
    """Convenience constructor matching Lab10's ``model = FlaxModel(...)`` pattern."""
    return WormComposer(hidden=hidden, depth=depth, out_dim=k_chopin)


def init_params(model: WormComposer, k_worm: int,
                 seed: int = 0) -> dict:
    """Initialise parameters via a dummy forward pass (Lab10 pattern).

    jax.random.PRNGKey is required for all JAX randomness (Lab04+).
    """
    key   = jax.random.PRNGKey(seed)
    dummy = jnp.zeros((1, k_worm))
    return model.init(key, dummy)


def mse_loss(params: dict, model: WormComposer,
             Z_batch: jnp.ndarray, C_batch: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error between predicted and target Chopin features.

    L14 connection: jax.value_and_grad applied to THIS function gives the
    gradient of the loss with respect to all network parameters in one pass —
    this is reverse-mode autodiff (backpropagation) in disguise.
    """
    C_pred = model.apply(params, Z_batch)
    return jnp.mean((C_pred - C_batch) ** 2)


def predict(params: dict, model: WormComposer,
             Z: np.ndarray) -> np.ndarray:
    """Run inference (no gradient tracking needed)."""
    return np.array(model.apply(params, jnp.array(Z, dtype=jnp.float32)))


def model_summary(model: WormComposer, k_worm: int) -> str:
    """Print parameter count and architecture."""
    params = init_params(model, k_worm)
    n_params = sum(v.size for v in jax.tree_util.tree_leaves(params))
    lines = [
        f"WormComposer: {k_worm} → [{model.hidden}]×{model.depth} → {model.out_dim}",
        f"Total parameters: {n_params:,}",
        f"Activation: tanh  |  Init: Xavier normal  |  Framework: JAX/Flax",
    ]
    return "\n".join(lines)
