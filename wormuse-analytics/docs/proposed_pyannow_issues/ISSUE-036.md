## ISSUE-036 — Step 8 PINN is oscillator-PINN, not the advertised ion-channel PINN `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | ✅ Partially Resolved — v0.8.0 (doc note added; HH PINN still not implemented) |
| **Priority** | P2 |
| **Severity** | Architecture / honesty — the "PINN centerpiece" claim does not match the code |

**Description.** `ION_CHANNELS.md` advertises an ion-channel (Hodgkin-Huxley) PINN as the project centerpiece. But `step8_pinn/locomotion_pinn.py` implements a **damped harmonic oscillator ODE** and **1D wave equation PDE** — locomotion physics, not ion-channel kinetics.

**Fix (v0.8.0).** Added a prominent architecture note box to the top of `locomotion_pinn.py`:

```
╔══════════════════════════════════════════════════════════════╗
║  ARCHITECTURE NOTE (ISSUE-036, v0.8.0)                       ║
║  This module enforces a LOCOMOTION-OSCILLATOR ODE/PDE.        ║
║  It is NOT the ion-channel PINN described in ION_CHANNELS.md. ║
║  The HH PINN is tracked in ISSUE-008 (ca_thresh PINN param). ║
╚══════════════════════════════════════════════════════════════╝
```

**Remaining work.** The ion-channel HH PINN (learning m_∞, h_∞, τ_m, τ_h from gating-variable trajectories) is not yet implemented. Track in ISSUE-008.

**AppStat connection.** Naming and framing matter for scientific honesty. The locomotion-oscillator PINN is a valid surrogate model — it just needs to be called what it is.

**Category:** `Category C — Architecture Clarity`
