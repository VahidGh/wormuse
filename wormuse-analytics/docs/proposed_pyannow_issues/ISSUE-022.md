## ISSUE-022 — Descriptive statistics of per-step outputs missing `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P2 |
| **Severity** | Interpretability — metric paradox is argued from formulas, never visualised |

**Description.** Notebook 03 prints scalar numbers (`len(onsets)`, `loss`, `F1`, `precision`, `recall`, `IOI`) per step but never builds the *distribution* view that AppStat Lecture 00 / Lab I uses as its foundation. A reader who plotted the inter-onset-interval (IOI) histograms side-by-side would immediately see that Step 0's IOIs cluster at one body-wave period while Chopin's are broad — the metric paradox of ISSUE-016 becomes obvious. Currently nothing in notebook 03 produces this view.

**Fix plan.**

1. Add a new cell after the data-preparation block (cell 4):

   ```python
   import pandas as pd
   from scipy.stats import kurtosis, skew

   def step_summary(onsets, label):
       o = np.sort(onsets[~np.isnan(onsets)])
       if len(o) < 2:
           return {'label': label, 'n': len(o),
                   'ioi_mean': np.nan, 'ioi_std': np.nan,
                   'ioi_skew': np.nan, 'ioi_kurtosis': np.nan}
       ioi = np.diff(o)
       return {'label': label, 'n': len(o),
               'ioi_mean': ioi.mean(), 'ioi_median': np.median(ioi),
               'ioi_std': ioi.std(ddof=1),
               'ioi_skew': skew(ioi), 'ioi_kurtosis': kurtosis(ioi),
               'ioi_min': ioi.min(), 'ioi_max': ioi.max()}
   ```

   Compute and collect into a single DataFrame as each step's onsets become available.

2. Add a cell that overlays KDEs of each step's IOI distribution against Chopin's:

   ```python
   import seaborn as sns
   fig, ax = plt.subplots(figsize=(9, 4))
   for name, ons in steps_dict.items():
       ioi = np.diff(np.sort(ons))
       sns.kdeplot(ioi[ioi < 1.5], ax=ax, fill=True, alpha=.15, label=name)
   ioi_c = np.diff(np.sort(t_on_chopin[t_on_chopin <= DURATION]))
   sns.kdeplot(ioi_c[ioi_c < 1.5], ax=ax, color='black', ls='--', lw=2, label='Chopin')
   ax.set(xlabel='IOI (s)', ylabel='density',
          title='Inter-onset interval distributions per step (Lab I)')
   ax.legend(fontsize=8); plt.tight_layout(); plt.show()
   ```

3. Reference implementation: `wormuse-analytics/src/wormuse_analytics/descriptive.py` — `step_summary`, `collect_step_stats`, `plot_ioi_distributions`.

**Affected files.**
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — new cells (Lab I block).
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- Optional: `PyANNOW/src/pyannow/targets/midi_target.py` — promote `step_summary` to a public helper.
- `PyANNOW/TODO.md` — this entry.
