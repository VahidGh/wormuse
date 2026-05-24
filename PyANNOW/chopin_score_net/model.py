"""Flax model for time-conditioned Chopin score reconstruction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class ResidualBlock(nn.Module):
    width: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.width)(x)
        h = nn.gelu(h)
        h = nn.LayerNorm()(h)
        h = nn.Dense(self.width)(h)
        h = nn.gelu(h)
        return x + h


class TimeScoreNet(nn.Module):
    """Residual MLP that maps Fourier time features to pitch logits."""

    n_pitches: int
    width: int = 256
    depth: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.width)(x)
        h = nn.gelu(h)
        h = nn.LayerNorm()(h)

        for _ in range(self.depth):
            h = ResidualBlock(self.width)(h)

        h = nn.Dense(self.width // 2)(h)
        h = nn.gelu(h)
        return nn.Dense(self.n_pitches)(h)


def build_model(n_features: int, n_pitches: int, width: int = 256, depth: int = 3) -> tuple[TimeScoreNet, dict]:
    """Initialise the model and its parameters."""

    model = TimeScoreNet(n_pitches=n_pitches, width=width, depth=depth)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, n_features), dtype=jnp.float32))
    return model, params


def score_logits(model: TimeScoreNet, params: dict, features: np.ndarray) -> np.ndarray:
    """Run the model in inference mode."""

    x = jnp.asarray(features, dtype=jnp.float32)
    return np.asarray(model.apply(params, x))


def weighted_bce_with_logits(logits: jnp.ndarray, targets: jnp.ndarray, pos_weight: jnp.ndarray | None = None) -> jnp.ndarray:
    """Binary cross entropy with optional positive-class weighting."""

    targets = jnp.asarray(targets, dtype=jnp.float32)
    logits = jnp.asarray(logits, dtype=jnp.float32)
    if pos_weight is None:
        pos_weight = jnp.ones((targets.shape[-1],), dtype=jnp.float32)
    else:
        pos_weight = jnp.asarray(pos_weight, dtype=jnp.float32)

    pos_term = pos_weight * targets * jax.nn.softplus(-logits)
    neg_term = (1.0 - targets) * jax.nn.softplus(logits)
    return jnp.mean(pos_term + neg_term)
