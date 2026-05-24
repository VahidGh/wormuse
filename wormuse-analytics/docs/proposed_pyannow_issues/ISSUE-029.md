## ISSUE-029 — Synthetic 302-neuron matrix masquerades as real neural data `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P1 |
| **Severity** | Correctness — every step downstream of this operates on synthetic data |

**Description.** In `_build_naml_progression_nb.py` cell 4 (and notebook 03 cell 4):

```python
n_neurons = 302
X_neural = np.zeros((n_neurons, T_pts))
for i in range(n_neurons):
    muscle_idx = i % n_muscles                       # ← repeats 8 muscles 38× each
    noise_amp  = 0.05 * (1 + i / n_neurons)
    X_neural[i] = V_mus[:, muscle_idx] + rng.standard_normal(T_pts) * noise_amp
```

The "302 neurons" are 8 muscle voltages replicated 38 times with Gaussian noise of amplitude ≤ 0.05× the signal. The matrix's **effective rank is 8** (or less). Every step downstream that claims to "compress neural activity" — Step 1a's RSVD to k=4, Step 2's PCA + KMeans — is operating on this synthetic block structure, *not* on a 302-D neural manifold.

**Proof.** Run:

```python
sv = np.linalg.svd(X_neural - X_neural.mean(axis=1, keepdims=True), compute_uv=False)
var = sv**2 / (sv**2).sum()
print(f'cumvar at k=8: {np.cumsum(var)[7]:.4f}')   # ≈ 0.999+
```

Or import the reference: `wormuse_analytics.pipeline.diagnose_synthetic_neural`.

**Fix plan — three options, in increasing rigor.**

1. **Document the limitation honestly** (cheapest). Rename `X_neural` to `X_neural_synthetic` and add a markdown cell stating: "the 302-D matrix is a synthetic block expansion of 8 muscle voltages; the PCA below trivially recovers the 8 muscle directions plus noise." Step 1a's narrative ("compress 302 → 4") becomes "compress 8 noisy muscle signals → 4". Honest and small change.

2. **Run PCA / RSVD directly on the 95-muscle matrix** (medium). `V_muscles ∈ ℝ^{T × 95}` is **real biological data** from the forward model. Compressing it to k_w with `k_w = choose_k_by_variance(V_mus, 0.9)` gives a meaningful neural-state encoder. All downstream steps stay the same.

3. **Bring real C302 spike data** (already open as ISSUE-010 in PyANNOW; out of scope for this audit). The `wormuse-sim/src/ow_bridge/` stub would need to be implemented; until then, option 2 is the right move.

**Recommended.** Option 2 — small change, big honesty win.

**Affected files.**
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` cell 4 — replace `X_neural` with `V_muscles.T`.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/docs/PyANNOW_NAML_progression.md` — update Step 1a narrative.
- `PyANNOW/TODO.md` — this entry.
