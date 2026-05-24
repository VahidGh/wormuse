## ISSUE-021 — Per-step F1 has no confidence interval; rank claims unsupported `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P2 |
| **Severity** | Statistical validity — point-estimate comparisons have no error bar |

**Description.** Each step's F1 is computed on the full 10 s window and reported as a single number. Claims like "Step 6 > Step 0" or "MLP+L-BFGS is the winner" therefore have no statistical content — we don't know if the difference would survive a different sub-window of Chopin, or if it falls inside the noise. The AppStat-correct answer is a **bootstrap confidence interval** per step and a **paired-bootstrap hypothesis test** for pairwise step comparisons.

**Fix plan.** Add to `pyannow/targets/midi_target.py`:

```python
def bootstrap_f1(worm_onsets, target_onsets, B=1000,
                 window_s=15.0, sub_window_s=5.0, tol_s=0.05,
                 random_state=0) -> dict:
    """Bootstrap CIs by resampling sub-windows.  Returns median + 2.5/97.5 percentiles."""
    rng = np.random.default_rng(random_state)
    f1s = []
    for _ in range(B):
        t0 = rng.uniform(0.0, max(0.0, window_s - sub_window_s))
        w = worm_onsets[(worm_onsets >= t0) & (worm_onsets <= t0 + sub_window_s)] - t0
        t = target_onsets[(target_onsets >= t0) & (target_onsets <= t0 + sub_window_s)] - t0
        if len(w) == 0 or len(t) == 0:
            f1s.append(0.0); continue
        f1s.append(musical_f1(w, t, tol_s=tol_s, window_s=sub_window_s)['f1'])
    f1s = np.array(f1s)
    return {'median': float(np.median(f1s)),
            'ci_low':  float(np.percentile(f1s, 2.5)),
            'ci_high': float(np.percentile(f1s, 97.5)),
            'samples': f1s}

def paired_bootstrap_compare(worm_a, worm_b, target_onsets, B=1000, ...):
    """H0: F1(a) == F1(b).  Returns median_diff + one-sided p-value + 95% CI."""
    # ... as in wormuse-analytics/src/wormuse_analytics/metrics.py
```

Reference implementation is in `wormuse-analytics/src/wormuse_analytics/metrics.py` (`bootstrap_f1`, `paired_bootstrap_compare`).

In notebook 03 cell 24, add two columns to the summary table:

| Step | F1 | F1 CI low | F1 CI high | IOI sim |

And the paired-bootstrap test for `H0: F1(step_k) ≤ F1(step_0)` so the "Step 0 wins" question gets a formal answer.

**Affected files.**
- `PyANNOW/src/pyannow/targets/midi_target.py` — add two functions.
- `PyANNOW/tests/test_midi_target.py` — bootstrap reproducibility test (seeded).
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — cell 24 augmented summary table.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/TODO.md` — this entry.
