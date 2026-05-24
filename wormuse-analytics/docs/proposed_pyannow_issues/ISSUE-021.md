## ISSUE-021 — No bootstrap confidence intervals on F1 `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | ✅ Resolved — v0.8.0 |
| **Priority** | P2 |
| **Severity** | Statistical validity — point estimates without uncertainty are uninterpretable |

**Description.** All per-step F1 scores were single point estimates computed on one fixed 10s window with no CI. It was impossible to tell whether Step 3 F1=0.18 vs Step 4 F1=0.19 was a meaningful improvement or noise.

**Fix (v0.8.0).** Two complementary bootstrap functions:

1. **`bootstrap_musical_f1()`** in `midi_target.py` — resamples Chopin target onsets with replacement (non-parametric CI on F1 sensitivity to which notes are evaluated):

```python
r = bootstrap_musical_f1(worm_onsets, chopin_onsets, n_boot=500, seed=42)
print(f"F1: {r['mean_f1']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
```

2. **`blocked_bootstrap_ci()`** in `training/cv.py` — block-bootstrap CI on any fit+score pair (preserves temporal autocorrelation structure — see ISSUE-038):

```python
bb = blocked_bootstrap_ci(fit_fn, score_fn, Z, C, block_len=20, n_boot=200)
```

**Threshold for significance.** Steps are significantly different if their 95% CIs do not overlap. Step 0 CI can be used as the baseline; all other steps must have CI entirely above Step 0's CI upper bound to claim improvement.

**Tested.** `test_midi_target.py::TestBootstrapMusicalF1` — CI contains mean, perfect-match mean > 0.50, empty input → zeros.

**AppStat connection.** L06: the bootstrap is the standard non-parametric method for CI estimation when the analytical form of the sampling distribution is unknown.

**Category:** `Category A — Metrics & Scoring`
