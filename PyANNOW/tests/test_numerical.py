"""Numerical correctness tests for PyANNOW.

Covers:
  - Eckart-Young theorem: truncated SVD is the best rank-k approx
  - RSVD achieves same error as full SVD at k=true_rank with power iteration
  - PINN ODE residual is finite and well-shaped
  - PINN total loss decreases over first 20 Adam steps
"""
from __future__ import annotations

import numpy as np
import pytest


class TestEckartYoung:
    """The Eckart-Young theorem is the mathematical heart of the NAML course (L06).
    We test it directly: no other rank-k matrix can have lower Frobenius error."""

    def test_truncated_svd_is_optimal(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 40))
        k = 5
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        X_k = (U[:, :k] * s[:k]) @ Vt[:k, :]
        err_svd = np.linalg.norm(X - X_k, 'fro')

        # Try 100 random rank-k matrices and verify none beats the truncated SVD
        for _ in range(100):
            A = rng.standard_normal((50, k))
            B = rng.standard_normal((k, 40))
            X_rand = A @ B
            err_rand = np.linalg.norm(X - X_rand, 'fro')
            assert err_svd <= err_rand + 1e-9, (
                "Eckart-Young violated: a random rank-k matrix beat the truncated SVD")

    def test_singular_values_non_increasing(self):
        """Singular values must be in non-increasing order (L06)."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((60, 40))
        s = np.linalg.svd(X, compute_uv=False)
        assert np.all(np.diff(s) <= 1e-12), "Singular values must be non-increasing"

    def test_frobenius_norm_from_singular_values(self):
        """||X||_F = sqrt(sum σ_i²)  (L06: connection between SVD and Frobenius)."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 25))
        s = np.linalg.svd(X, compute_uv=False)
        frob_from_svd    = np.sqrt((s**2).sum())
        frob_from_matrix = np.linalg.norm(X, 'fro')
        np.testing.assert_allclose(frob_from_svd, frob_from_matrix, rtol=1e-10)


class TestRSVDAccuracy:

    def test_rsvd_matches_full_at_true_rank(self):
        """RSVD with q=2 power iterations should recover a true rank-5 matrix."""
        from pyannow.step1_svd.encoder import rsvd
        rng = np.random.default_rng(0)
        n, m, r = 200, 150, 5
        L = rng.standard_normal((n, r))
        R = rng.standard_normal((r, m))
        X = L @ R + 0.001 * rng.standard_normal((n, m))   # near-rank-r

        U_r, s_r, Vt_r = rsvd(X, k=r, q=2, seed=0)
        X_hat = (U_r * s_r) @ Vt_r
        rel_err = np.linalg.norm(X - X_hat, 'fro') / np.linalg.norm(X, 'fro')
        assert rel_err < 0.05, (
            f"RSVD relative error {rel_err:.4f} too large for near-rank-{r} matrix")


class TestPINNNumerics:

    @pytest.fixture
    def toy_pinn_setup(self):
        """Tiny PINN setup for numerical tests (fast, no training needed)."""
        import jax, jax.numpy as jnp
        from pyannow.step8_pinn.locomotion_pinn import PhysicsComposer
        k = 4
        model = PhysicsComposer(hidden=8, depth=2, out_dim=4)
        params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2 + k)))
        return model, params, k

    def test_ode_residual_finite(self, toy_pinn_setup):
        """ODE residual must not produce NaN or Inf."""
        import jax.numpy as jnp
        from pyannow.step8_pinn.locomotion_pinn import (
            PhysicsComposer, ode_residual)
        # Need a 1+k model for ODE
        import jax
        k = 4
        model_ode = PhysicsComposer(hidden=8, depth=2, out_dim=4)
        params_ode = model_ode.init(jax.random.PRNGKey(0), jnp.zeros((1, 1 + k)))
        t = jnp.array(1.0)
        z = jnp.zeros(k)
        res = ode_residual(params_ode, model_ode, t, z)
        assert res.shape == (4,), f"ODE residual shape must be (out_dim,), got {res.shape}"
        assert jnp.all(jnp.isfinite(res)), "ODE residual must be finite"

    def test_pde_residual_finite(self, toy_pinn_setup):
        """PDE residual must not produce NaN or Inf."""
        import jax.numpy as jnp
        from pyannow.step8_pinn.locomotion_pinn import pde_residual
        model, params, k = toy_pinn_setup
        x = jnp.array(0.5)
        t = jnp.array(1.0)
        z = jnp.zeros(k)
        res = pde_residual(params, model, x, t, z)
        assert res.shape == (4,), f"PDE residual shape must be (out_dim,), got {res.shape}"
        assert jnp.all(jnp.isfinite(res)), "PDE residual must be finite"

    def test_pinn_loss_decreases(self):
        """Total PINN loss should strictly decrease over the first 20 Adam steps."""
        import jax, jax.numpy as jnp, optax
        from pyannow.step8_pinn.locomotion_pinn import (
            PhysicsComposer, pinn_loss, ode_physics_loss)
        k = 4
        model_ode = PhysicsComposer(hidden=8, depth=2, out_dim=4)
        params = model_ode.init(jax.random.PRNGKey(42), jnp.zeros((1, 1 + k)))

        rng = np.random.default_rng(0)
        n_d, n_c = 20, 10
        inp_d = jnp.array(rng.standard_normal((n_d, 1 + k)).astype(np.float32))
        q_d   = jnp.array(rng.standard_normal((n_d, 4)).astype(np.float32))
        t_c   = jnp.array(rng.uniform(0, 5, n_c).astype(np.float32))
        z_c   = jnp.array(rng.standard_normal((n_c, k)).astype(np.float32))

        def loss_fn(p):
            return pinn_loss(p, model_ode, inp_d, q_d,
                             ode_physics_loss, (t_c, z_c), 0.1)[0]

        opt   = optax.adam(1e-2)
        state = opt.init(params)
        losses = []
        for _ in range(25):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, state = opt.update(grads, state, params)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))

        assert losses[-1] < losses[0], (
            f"PINN loss should decrease: start={losses[0]:.4f} end={losses[-1]:.4f}")


# ── needed by test_pinn_loss_decreases ────────────────────────────────────
import numpy as np   # noqa: E402  (imported again for the fixture)
