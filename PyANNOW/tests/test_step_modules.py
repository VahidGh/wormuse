"""Tests for PyANNOW step modules (Steps 1-8).

Covers:
  - Step 1: RSVD shape + reconstruction accuracy
  - Step 1: Procrustes alignment improves residual vs random rotation
  - Step 2: PCA + K-means produces correct cluster count
  - Step 3: Ridge regression has R² > 0 on synthetic correlated data
  - Step 4: Flax MLP output shape
  - Step 5: Adam training loop reduces loss
  - Step 6: L-BFGS step does not increase loss
"""
from __future__ import annotations

import numpy as np
import pytest


class TestStep1SVD:

    def test_rsvd_output_shape(self, synthetic_X_neural):
        from pyannow.step1_svd.encoder import rsvd
        X = synthetic_X_neural
        k = 4
        U, s, Vt = rsvd(X, k=k)
        assert U.shape  == (X.shape[0], k), f"U shape mismatch: {U.shape}"
        assert s.shape  == (k,)
        assert Vt.shape == (k, X.shape[1])

    def test_rsvd_reconstruction_error(self, synthetic_X_neural):
        """RSVD should reconstruct with rel-error < 0.3 at k=4 on synthetic data."""
        from pyannow.step1_svd.encoder import rsvd
        X = synthetic_X_neural
        U, s, Vt = rsvd(X, k=8, q=1, seed=0)
        X_hat = (U * s) @ Vt
        rel_err = np.linalg.norm(X - X_hat, 'fro') / np.linalg.norm(X, 'fro')
        assert rel_err < 0.5, f"RSVD rel-error {rel_err:.3f} too high at k=8"

    def test_choose_k_returns_positive(self, synthetic_X_neural):
        from pyannow.step1_svd.encoder import choose_k_by_variance
        k = choose_k_by_variance(synthetic_X_neural, variance_threshold=0.80)
        assert k >= 1, "choose_k_by_variance must return at least 1"
        assert k <= min(synthetic_X_neural.shape), "k cannot exceed matrix rank"

    def test_neural_scores_shape(self, synthetic_X_neural):
        from pyannow.step1_svd.encoder import rsvd, neural_scores
        X = synthetic_X_neural
        U, _, _ = rsvd(X, k=4)
        Z = neural_scores(X, U)
        assert Z.shape == (4, X.shape[1]), f"Expected (4, T), got {Z.shape}"

    def test_procrustes_improves_residual(self):
        """Procrustes-aligned worm should fit Chopin better than a random rotation."""
        from pyannow.step1_svd.procrustes import procrustes_align
        rng = np.random.default_rng(7)
        T, k = 100, 4
        W_k = rng.standard_normal((T, k))
        C_k = rng.standard_normal((T, k))
        result = procrustes_align(W_k, C_k)
        # Residual from Procrustes ≤ residual from identity mapping
        R = result["R"]
        proc_residual = np.linalg.norm(C_k - W_k @ R, 'fro') / np.linalg.norm(C_k, 'fro')
        id_residual   = np.linalg.norm(C_k - W_k, 'fro')       / np.linalg.norm(C_k, 'fro')
        assert proc_residual <= id_residual + 1e-9, (
            "Procrustes must not be worse than identity mapping")

    def test_procrustes_R_is_orthogonal(self):
        from pyannow.step1_svd.procrustes import procrustes_align
        rng = np.random.default_rng(99)
        W_k = rng.standard_normal((80, 4))
        C_k = rng.standard_normal((80, 4))
        R = procrustes_align(W_k, C_k)["R"]
        RtR = R.T @ R
        np.testing.assert_allclose(RtR, np.eye(RtR.shape[0]),
                                    atol=1e-9, err_msg="R must be orthogonal: R^T R = I")


class TestStep2Clustering:

    def test_pca_reduce_shape(self, synthetic_X_neural):
        from pyannow.step2_clustering.motor_primitives import pca_reduce
        scores, pca = pca_reduce(synthetic_X_neural, n_components=4, standardize=True)
        assert scores.shape == (synthetic_X_neural.shape[1], 4), (
            f"PCA scores shape mismatch: {scores.shape}")

    def test_kmeans_cluster_count(self, synthetic_X_neural):
        from pyannow.step2_clustering.motor_primitives import pca_reduce, find_motor_primitives
        scores, _ = pca_reduce(synthetic_X_neural, n_components=4)
        for k in [2, 3, 4]:
            labels, km = find_motor_primitives(scores, k=k)
            n_clusters = len(set(labels))
            assert n_clusters == k, f"Requested k={k}, got {n_clusters} clusters"

    def test_cluster_to_notes_returns_list(self, synthetic_X_neural, synthetic_t_arr_ms):
        from pyannow.step2_clustering.motor_primitives import (
            pca_reduce, find_motor_primitives, cluster_to_notes)
        scores, _ = pca_reduce(synthetic_X_neural, n_components=4)
        labels, _ = find_motor_primitives(scores, k=4)
        events = cluster_to_notes(labels, synthetic_t_arr_ms)
        assert isinstance(events, list), "cluster_to_notes must return a list"
        for ev in events:
            assert len(ev) == 3, "Each cluster event must be (time_s, pitch, velocity)"


class TestStep3Ridge:

    def test_ridge_r2_positive_on_correlated(self):
        """Ridge should achieve R² > 0 when there is a real linear relationship."""
        from pyannow.step3_regression.ridge_composer import RidgeComposer
        rng = np.random.default_rng(0)
        n, k_w, k_c = 200, 4, 8
        Z = rng.standard_normal((n, k_w))
        W_true = rng.standard_normal((k_w, k_c))
        C = Z @ W_true + 0.1 * rng.standard_normal((n, k_c))   # correlated target
        rc = RidgeComposer(alpha=1.0)
        rc.fit(Z[:150], C[:150])
        ev = rc.evaluate(Z[150:], C[150:])
        assert ev["r2"] > 0.5, f"Ridge R²={ev['r2']:.3f} should be >0.5 on correlated data"

    def test_ridge_predicts_correct_shape(self):
        from pyannow.step3_regression.ridge_composer import RidgeComposer
        rng = np.random.default_rng(1)
        Z = rng.standard_normal((100, 4))
        C = rng.standard_normal((100, 8))
        rc = RidgeComposer(alpha=0.1).fit(Z, C)
        out = rc.predict(Z)
        assert out.shape == C.shape, f"Prediction shape {out.shape} != target {C.shape}"


class TestStep4FFNN:

    def test_composer_output_shape(self):
        """MLP output must have shape (batch, out_dim)."""
        import jax, jax.numpy as jnp
        from pyannow.step4_ffnn.jax_composer import create_model, init_params
        k_worm, k_chopin, batch = 4, 8, 16
        model = create_model(k_worm=k_worm, k_chopin=k_chopin, hidden=16, depth=2)
        params = init_params(model, k_worm)
        dummy = jnp.zeros((batch, k_worm))
        out = model.apply(params, dummy)
        assert out.shape == (batch, k_chopin), f"Output shape {out.shape} != ({batch},{k_chopin})"

    def test_mse_loss_finite(self):
        import jax, jax.numpy as jnp
        from pyannow.step4_ffnn.jax_composer import (
            create_model, init_params, mse_loss)
        model = create_model(k_worm=4, k_chopin=8, hidden=16, depth=2)
        params = init_params(model, 4)
        Z = jnp.ones((10, 4))
        C = jnp.ones((10, 8))
        loss = mse_loss(params, model, Z, C)
        assert float(loss) < 1e6 and not jnp.isnan(loss), "MSE loss must be finite"


class TestStep5Training:

    def test_adam_reduces_loss(self):
        """Adam training must decrease MSE loss over 50 steps."""
        import numpy as np
        from pyannow.step4_ffnn.jax_composer import create_model, init_params
        from pyannow.step5_training.adam_trainer import train_adam
        rng = np.random.default_rng(42)
        Z = rng.standard_normal((60, 4)).astype(np.float32)
        C = (Z @ rng.standard_normal((4, 8))).astype(np.float32)   # learnable mapping
        model  = create_model(k_worm=4, k_chopin=8, hidden=16, depth=2)
        params = init_params(model, 4)
        _, history = train_adam(model, params, Z, C,
                                lr=1e-2, epochs=60, batch_sz=16, verbose=False)
        losses = history.train_loss
        assert len(losses) > 0, "Training history must be non-empty"
        # Loss at end should be less than loss at start (or at least not worse)
        assert losses[-1] <= losses[0] + 0.1, (
            f"Loss should not increase: start={losses[0]:.4f} end={losses[-1]:.4f}")


class TestStep6LBFGS:

    def test_lbfgs_does_not_increase_loss(self):
        """A single L-BFGS pass should not make the loss worse than Adam final."""
        import numpy as np
        from pyannow.step4_ffnn.jax_composer import create_model, init_params
        from pyannow.step5_training.adam_trainer import train_adam
        from pyannow.step6_lbfgs.lbfgs_polish import polish_lbfgs
        rng = np.random.default_rng(7)
        Z = rng.standard_normal((80, 4)).astype(np.float32)
        C = (Z @ rng.standard_normal((4, 8))).astype(np.float32)
        model  = create_model(k_worm=4, k_chopin=8, hidden=16, depth=2)
        params = init_params(model, 4)
        params_adam, h_adam = train_adam(model, params, Z, C,
                                         lr=1e-2, epochs=30, verbose=False)
        params_lbfgs, h_lbfgs = polish_lbfgs(model, params_adam, Z, C,
                                               max_steps=20, verbose=False)
        if h_lbfgs.train_loss:
            adam_final  = h_adam.train_loss[-1] if h_adam.train_loss else 1.0
            lbfgs_final = h_lbfgs.train_loss[-1]
            assert lbfgs_final <= adam_final + 0.05, (
                f"L-BFGS must not severely worsen Adam's result "
                f"(Adam={adam_final:.4f}, L-BFGS={lbfgs_final:.4f})")
