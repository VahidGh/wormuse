"""Training helpers for the Chopin score model."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data import best_frame_threshold
from .model import TimeScoreNet, build_model, weighted_bce_with_logits


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)


def _split_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]


def train_score_model(
    features: np.ndarray,
    target_roll: np.ndarray,
    width: int = 256,
    depth: int = 3,
    lr: float = 1e-3,
    epochs: int = 250,
    batch_size: int = 512,
    val_frac: float = 0.15,
    patience: int = 25,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[TimeScoreNet, dict, TrainHistory]:
    """Fit the time-conditioned score model with Adam and early stopping."""

    features = np.asarray(features, dtype=np.float32)
    target_roll = np.asarray(target_roll, dtype=np.float32)

    tr_idx, va_idx = _split_indices(len(features), val_frac=val_frac, seed=seed)
    x_tr = jnp.asarray(features[tr_idx], dtype=jnp.float32)
    y_tr = jnp.asarray(target_roll[tr_idx], dtype=jnp.float32)
    x_va = jnp.asarray(features[va_idx], dtype=jnp.float32)
    y_va = jnp.asarray(target_roll[va_idx], dtype=jnp.float32)

    model, params = build_model(features.shape[1], target_roll.shape[1], width=width, depth=depth)

    positives = target_roll.sum(axis=0)
    negatives = max(1.0, float(target_roll.shape[0])) - positives
    pos_weight = jnp.asarray((negatives + 1.0) / (positives + 1.0), dtype=jnp.float32)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step_fn(p, s, xb, yb):
        loss_fn = lambda pp: weighted_bce_with_logits(model.apply(pp, xb), yb, pos_weight=pos_weight)
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, s = optimizer.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss

    @jax.jit
    def loss_fn(p, xb, yb):
        return weighted_bce_with_logits(model.apply(p, xb), yb, pos_weight=pos_weight)

    rng = np.random.default_rng(seed)
    history = TrainHistory()
    best_val = float("inf")
    best_params = params
    no_improve = 0

    n_train = len(tr_idx)
    n_batches = max(1, int(np.ceil(n_train / batch_size)))

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n_train)
        tr_epoch = 0.0

        for b in range(n_batches):
            batch = perm[b * batch_size : (b + 1) * batch_size]
            xb = x_tr[batch]
            yb = y_tr[batch]
            params, opt_state, loss = step_fn(params, opt_state, xb, yb)
            tr_epoch += float(loss)

        tr_epoch /= n_batches
        val_loss = float(loss_fn(params, x_va, y_va))
        history.epochs.append(epoch)
        history.train_loss.append(tr_epoch)
        history.val_loss.append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch == 1 or epoch % 25 == 0):
            print(f"epoch {epoch:3d}  train={tr_epoch:.5f}  val={val_loss:.5f}")

        if no_improve >= patience:
            if verbose:
                print(f"early stop at epoch {epoch}  val={val_loss:.5f}")
            break

    return model, best_params, history


def predict_probabilities(model: TimeScoreNet, params: dict, features: np.ndarray) -> np.ndarray:
    """Return pitch probabilities for each time step."""

    logits = np.asarray(model.apply(params, jnp.asarray(features, dtype=jnp.float32)))
    return 1.0 / (1.0 + np.exp(-logits))


def best_threshold(probs: np.ndarray, target_roll: np.ndarray) -> tuple[float, float]:
    """Select a threshold for event reconstruction."""

    return best_frame_threshold(probs, target_roll)
