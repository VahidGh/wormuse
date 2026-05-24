## ISSUE-036 — Step 8 PINN is oscillator-PINN, not the advertised ion-channel PINN `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P2 |
| **Severity** | Architecture / honesty — the "PINN centerpiece" claim does not match the code |

**Description.** `ION_CHANNELS.md` advertises:

> The whole project pivots on this: **ion channels tune the worm's neural firing, the firing triggers the piano, and the piano makes music.** A Physics-Informed Neural Network (PINN) learns the channel kinetics that make the worm musical.

But `pyannow/step8_pinn/locomotion_pinn.py` (Step 8) trains a PINN on a **damped harmonic oscillator ODE** (`q̈ + 2γq̇ + ω²q = F`) and on a **1-D wave equation** (`ρq_tt − μq_xx + γq_t = F`), with Chopin features as the data target. The physics being enforced is the *locomotion oscillator*, not the *Hodgkin-Huxley gating kinetics*.

Two consequences:

1. The "PINN centerpiece" claim is currently aspirational, not implemented.
2. Step 8 is hard to compare with Steps 0-6 because it minimises a different loss (data + physics on PCA-compressed Chopin features) on a different network family.

This was already acknowledged in PyANNOW ISSUE-017 ("Steps 8a/8b removed from `losses` dict — incommensurable"). But the documentation in ION_CHANNELS.md was not updated.

**Fix plan — two paths.**

1. **Rename and clarify** (cheap, honest). Rename `step8_pinn` to `step8_locomotion_pinn` and add a top-of-file note: "This module enforces a locomotion-oscillator ODE/PDE. The ion-channel HH PINN is the project's intended centerpiece (see ION_CHANNELS.md) but is not yet implemented; track in ISSUE-008."

2. **Implement the ion-channel PINN** (substantial work; the user has a separate SC-PINN repo). The HH PINN would learn `(m_∞, h_∞, τ_m, τ_h)` as functions of voltage from observed gating-variable trajectories. Loss = data + jax.grad-based residual of `dm/dt = (m_∞ - m)/τ_m`. This is what ION_CHANNELS.md describes; it doesn't exist in PyANNOW.

3. **Document the relationship between the two PINNs** in `docs/SCIENTIFIC_FOUNDATION.md`: the locomotion oscillator PINN is a *coarse-grained surrogate* for the dynamics that the ion-channel PINN would produce. It's a valid model in its own right; just not what the headline claims.

**Affected files.**
- `PyANNOW/src/pyannow/step8_pinn/locomotion_pinn.py` — top-of-file note; rename module to `step8_locomotion_pinn` (with deprecation alias).
- `PyANNOW/ION_CHANNELS.md` — disambiguate "the PINN" everywhere.
- `wormuse/docs/SCIENTIFIC_FOUNDATION.md` — section on the two PINNs.
- `PyANNOW/TODO.md` — this entry. Also reference ISSUE-008 (Ca_thresh as PINN parameter).
