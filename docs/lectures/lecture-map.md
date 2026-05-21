# Lecture → implementation map

Master table tying each lecture from the relevant Politecnico di Milano courses to a concrete file in wormuse. Filled in over time as code lands.

Convention: `□` = not yet implemented; `✓` = landed; `~` = partial.

---

## Advanced Methods for Scientific Computing (AMSC)

| Lecture               | Topic                                                         | Where in wormuse                                  | Status |
| --------------------- | ------------------------------------------------------------- | ------------------------------------------------- | ------ |
| 01 Introduction       | —                                                            | (no code)                                         | —     |
| 02 IntroFloats        | FP precision in time integration                              | `wormuse-sim/src/piano_sim/time_integrator.hpp` | □     |
| 03 IntroToCpp         | Modern C++17 baseline                                         | throughout `wormuse-sim`                        | □     |
| 04 Functions          | `[[nodiscard]]`, default args                               | `wormuse-sim/src/piano_sim/string_1d.hpp`       | □     |
| 05 SmartPointers      | `unique_ptr<WormSimulator>`                                 | `wormuse-sim/src/ow_bridge/docker_runner.hpp`   | □     |
| 06 Classes            | Virtual base `PianoString`, override                        | `wormuse-sim/src/piano_sim/string.hpp`          | □     |
| 07 ClassTemplates     | `PianoString<N_modes>`, traits, factory                     | `wormuse-sim/src/piano_sim/modal_string.hpp`    | □     |
| 08 StandardLibrary    | `std::variant`, `std::optional`, parallel `std::reduce` | `wormuse-sim/src/neuron_to_midi/events.hpp`     | □     |
| 09 Static/Shared libs | `wormuse_core` STATIC library + plugins                     | `wormuse-sim/CMakeLists.txt`                    | □     |
| 10 ParallelComputing  | Speedup analysis, Amdahl, Gustafson                           | `docs/design_notes/scaling.md`                  | □     |
| 11 MPI                | Distributed soundboard FEM (Phase 5+)                         | `wormuse-sim/src/piano_sim/dist_assembler.hpp`  | □     |
| 12 OpenMP             | Parallel loop over piano strings                              | `wormuse-sim/src/piano_sim/multi_string.cpp`    | □     |

## Numerical Linear Algebra (NLA)

| Lecture (Antonietti P0-P8)        | Concept                                          | Where                                               | Status |
| --------------------------------- | ------------------------------------------------ | --------------------------------------------------- | ------ |
| Direct methods (LU, QR, Cholesky) | Stiffness matrix factor in piano FEM             | `wormuse-sim/src/piano_sim/soundboard_fem.cpp`    | □     |
| Iterative (CG, GMRES, BiCGSTAB)   | Sparse solve for vibrating plate                 | `wormuse-sim/src/piano_sim/iterative_backend.cpp` | □     |
| Preconditioners (Jacobi, IC, AMG) | Speed up the piano CG                            | `wormuse-sim/src/piano_sim/preconditioner.hpp`    | □     |
| Power method, Lanczos             | Extract dominant modes of a string               | `wormuse-sim/src/piano_sim/modal_extractor.cpp`   | □     |
| LIS library                       | Alternative iterative backend                    | `wormuse-sim/src/piano_sim/lis_backend.cpp`       | □     |
| SVD                               | Eckart-Young low-rank truncation of neural state | `PyANNOW/src/pyannow/neural_state/svd_encoder.py` | □     |

## Numerical Methods for PDEs (NMPDE — Quarteroni)

| Topic                              | Concept                              | Where                                                             | Status |
| ---------------------------------- | ------------------------------------ | ----------------------------------------------------------------- | ------ |
| Weak form, Galerkin                | Soundboard plate weak formulation    | `docs/math_derivations/soundboard_weak_form.md`                 | □     |
| Lax-Milgram                        | Well-posedness of the plate equation | `docs/math_derivations/wellposedness.md`                        | □     |
| Céa's lemma + error estimate      | `O(h²)` convergence target        | `wormuse-sim/tests/test_convergence.cpp`                        | □     |
| θ-method for parabolic            | Time integration of damped plate     | `wormuse-sim/src/piano_sim/theta_method.hpp`                    | □     |
| FEM in 1D / 2D                     | String (1D) + soundboard (2D)        | `wormuse-sim/src/piano_sim/`                                    | □     |
| deal.II patterns                   | The 7-step solver class              | `wormuse-sim/src/piano_sim/soundboard_fem.hpp`                  | □     |
| Manufactured solution test         | Verification of FEM                  | `wormuse-sim/tests/test_manufactured.cpp`                       | □     |
| (Reuse) Heat.cpp CI infrastructure | CMake + CTest + scaling workflow     | `wormuse-sim/CMakeLists.txt` + `.github/workflows/verify.yml` | ~      |

## Parallel Computing (PC)

| Lecture                  | Concept                                  | Where                                                                                                       | Status |
| ------------------------ | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------ |
| L1-2 Intro / PRAM        | Theoretical baselines                    | `docs/design_notes/pram_analysis.md`                                                                      | □     |
| L3-4 Basic arch / models | (Background)                             | —                                                                                                          | —     |
| L5 GPU architecture      | CUDA backend for soundboard (stretch)    | `wormuse-sim/src/piano_sim/cuda/`                                                                         | □     |
| L6 Memory consistency    | Lock-free spike queue between threads    | `wormuse-sim/src/orchestrator/queue.hpp`                                                                  | □     |
| L7-7.75 OpenMP / POSIX   | `#pragma omp parallel for` for strings | `wormuse-sim/src/piano_sim/multi_string.cpp`                                                              | □     |
| L8-10 Patterns A/B/C     | map / reduce / scan / stencil            | `wormuse-sim/src/piano_sim/string_1d.cpp` (stencil), `wormuse-sim/src/orchestrator/energy.cpp` (reduce) | □     |

## Software Engineering for HPC (SE4HPC)

| Concept                                    | Where                                                            | Status |
| ------------------------------------------ | ---------------------------------------------------------------- | ------ |
| Verification workflow                      | `.github/workflows/verify.yml`                                 | ✓     |
| Scalability workflow (Phase 7+)            | `.github/workflows/scalability.yml`                            | □     |
| Containerization                           | `docker-compose.yml`                                           | ✓     |
| CI on push & PR                            | GitHub Actions setup                                             | ✓     |
| Reproducibility (fixed seeds, pinned deps) | `PyANNOW/pyproject.toml`, `wormuse-analytics/pyproject.toml` | □     |
| Modular CMake                              | `wormuse-sim/CMakeLists.txt` + per-module `cmake/`           | □     |

## Numerical Analysis for ML (NAML)

| Lecture                     | Concept                               | Where                                               | Status |
| --------------------------- | ------------------------------------- | --------------------------------------------------- | ------ |
| L02-05 Linear algebra       | Whitening, projection in encoder      | `PyANNOW/src/pyannow/neural_state/encoder.py`     | □     |
| L06 Eckart-Young            | Truncated SVD baseline encoder        | `PyANNOW/src/pyannow/neural_state/svd_encoder.py` | □     |
| L08 PCA                     | Diagnostic plots of neural latent     | `PyANNOW/src/pyannow/neural_state/diagnostics.py` | □     |
| L09 Pseudoinverse           | Decoder regression                    | `PyANNOW/src/pyannow/neural_state/decoder.py`     | □     |
| L14 Autodiff                | `jax.grad` for HH residual          | `PyANNOW/src/pyannow/ion_channels/pinn.py`        | □     |
| L15 Activations             | tanh for PINN                         | `PyANNOW/src/pyannow/ion_channels/pinn.py`        | □     |
| L16-17 NN basics            | Flax MLP architecture                 | `PyANNOW/src/pyannow/ion_channels/pinn.py`        | □     |
| L18-19 GD / SGD             | (Reference baseline)                  | `PyANNOW/src/pyannow/training/sgd.py`             | □     |
| L20 SGD variants (Adam)     | First-stage optimizer                 | `PyANNOW/src/pyannow/training/adam_stage.py`      | □     |
| L21 Newton                  | (Comparison only)                     | `docs/design_notes/newton_vs_lbfgs.md`            | □     |
| L22 L-BFGS                  | Second-stage polish (`optax.lbfgs`) | `PyANNOW/src/pyannow/training/lbfgs_stage.py`     | □     |
| L23 Convolution             | (Optional) conv composer              | `PyANNOW/src/pyannow/composer/conv_composer.py`   | □     |
| L24 Universal approximation | MLP capacity justification            | `docs/design_notes/architecture.md`               | □     |
| L25 Functional analysis     | Sobolev priors on `m_∞`, `τ_m`  | `docs/math_derivations/regularity.md`             | □     |
| L26 Complexity of NN        | Parameter budget                      | `docs/design_notes/capacity.md`                   | □     |
| **L27 PINNs**         | **The ion-channel PINN**        | `PyANNOW/src/pyannow/ion_channels/pinn.py`        | □     |

## Applied Statistics (AppStat)

| Lecture                             | Concept                                            | Where                                                       | Status |
| ----------------------------------- | -------------------------------------------------- | ----------------------------------------------------------- | ------ |
| 00 Intro                            | —                                                 | —                                                          | —     |
| 01 PCA                              | PCA on 302-D neural states                         | `wormuse-analytics/notebooks/Lab_II_PCA.ipynb`            | □     |
| 02 Nonlinear DR                     | t-SNE / UMAP of motor states                       | `wormuse-analytics/notebooks/Lab_II_PCA.ipynb`            | □     |
| 03 Clustering                       | KMeans / Ward / DBSCAN / GMM on motor primitives   | `wormuse-analytics/notebooks/Lab_III_clustering.ipynb`    | □     |
| 04 Linear models part 1 + 2         | OLS regression music-quality vs ion-channel params | `wormuse-analytics/notebooks/Lab_V_regression.ipynb`      | □     |
| 05 Logistic regression              | Musical vs noisy melody classifier                 | `wormuse-analytics/notebooks/Lab_VI_classification.ipynb` | □     |
| 06 Classification + model selection | CV, ROC, AUC                                       | `wormuse-analytics/notebooks/Lab_VI_classification.ipynb` | □     |
| 07 Tree-based                       | Random Forest baseline + feature importance        | `wormuse-analytics/notebooks/Lab_VI_classification.ipynb` | □     |

---

## Stretch goals (not in main roadmap)

| Course                  | Lecture        | Stretch idea                                                       |
| ----------------------- | -------------- | ------------------------------------------------------------------ |
| AppStat-R (old, Secchi) | LMM            | Repeated measurements per worm-genotype → LMM in R                |
| AppStat-R               | Geostats       | If we ever spatialize the analysis (worm-on-petri-dish) → kriging |
| AppStat-R               | FDA            | Treat each piano note as a functional response over time → FPCA   |
| NAML                    | (full SC-PINN) | Wire in the user's full SC-PINN as alternative backend             |

These map to `polimi-appstat-r` and the user's `naml-ion-channel-pinn/` external repos — kept out of the main roadmap per the "no submodule integration" decision.
