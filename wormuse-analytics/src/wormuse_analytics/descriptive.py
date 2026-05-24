"""Descriptive statistics — AppStat 2026 Lecture 00 / Lab I.

Lecture recap
-------------
The opening lecture and Lab I cover the foundation: summary statistics
(mean, median, variance, std, **skewness**, **kurtosis**), and visual
exploration (histograms, KDE, boxplots, pairplots).  The principle is
"plot first, then test."

For PyANNOW the relevant question is: *what does each step's note stream
look like distributionally?*  The metric paradox of ISSUE-016 is hard to
see from scalar loss numbers but obvious from IOI histograms:

- Chopin: broad IOI distribution, span 50 ms - 800 ms.
- Step 0:  near-Dirac spike at one body-wave period (~220 ms).
- A method that produced an IOI distribution looking like Chopin's would
  by necessity be musically better, regardless of what a noise-tolerant
  scalar metric reports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def step_summary(onsets: np.ndarray, label: str | None = None) -> dict:
    """Lab I summary statistics for one step's onset stream.

    Returns a dict with: n_notes, ioi_mean, ioi_median, ioi_std,
    ioi_skew, ioi_kurtosis, ioi_min, ioi_max.  Skewness uses Fisher's
    definition (0 = symmetric); kurtosis is excess (0 = normal).
    """
    onsets = np.asarray(onsets)
    onsets = np.sort(onsets[~np.isnan(onsets)])
    out = {"label": label, "n_notes": int(len(onsets))}
    if len(onsets) < 2:
        # Cannot compute IOI stats with < 2 notes
        out.update({"ioi_mean": np.nan, "ioi_median": np.nan, "ioi_std": np.nan,
                    "ioi_skew": np.nan, "ioi_kurtosis": np.nan,
                    "ioi_min": np.nan, "ioi_max": np.nan})
        return out
    ioi = np.diff(onsets)
    out.update({
        "ioi_mean":     float(ioi.mean()),
        "ioi_median":   float(np.median(ioi)),
        "ioi_std":      float(ioi.std(ddof=1)),
        "ioi_skew":     float(skew(ioi)),
        "ioi_kurtosis": float(kurtosis(ioi)),
        "ioi_min":      float(ioi.min()),
        "ioi_max":      float(ioi.max()),
    })
    return out


def collect_step_stats(steps: Iterable, chopin_onsets: np.ndarray | None = None) -> pd.DataFrame:
    """Build the Lab I summary DataFrame across all steps (+ Chopin row, if given).

    `steps` is iterable of either StepOutputs dataclasses or (label, onsets) tuples.
    """
    rows = []
    for s in steps:
        if hasattr(s, "name") and hasattr(s, "onsets"):
            rows.append(step_summary(s.onsets, label=s.name))
        else:
            label, onsets = s
            rows.append(step_summary(onsets, label=label))
    if chopin_onsets is not None:
        rows.append(step_summary(chopin_onsets, label="Chopin (target)"))
    return pd.DataFrame(rows).set_index("label")


def plot_ioi_distributions(steps, chopin_onsets, ax=None, max_ioi_s: float = 1.5):
    """Overlay KDE / histogram of IOI distribution per step + Chopin.

    Uses seaborn for the lab-style look.  Returns the matplotlib Axes.
    Each step's IOI distribution is plotted as a translucent density curve
    so the reader can see at a glance which steps approximate Chopin's
    rhythmic distribution.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))

    def _ioi(onsets):
        o = np.sort(onsets)
        return np.diff(o[~np.isnan(o)])

    for s in steps:
        ioi = _ioi(s.onsets) if hasattr(s, "onsets") else _ioi(s[1])
        label = s.name if hasattr(s, "name") else s[0]
        if len(ioi) < 2:
            continue
        ioi = ioi[ioi <= max_ioi_s]
        sns.kdeplot(ioi, ax=ax, label=label, fill=True, alpha=0.15, linewidth=1.5)

    ioi_c = _ioi(chopin_onsets)
    ioi_c = ioi_c[ioi_c <= max_ioi_s]
    sns.kdeplot(ioi_c, ax=ax, label="Chopin (target)", color="black",
                linewidth=2.0, linestyle="--", fill=False)

    ax.set(xlabel="Inter-onset interval (s)", ylabel="density",
           title="IOI distribution per step vs Chopin (Lab I)")
    ax.legend(fontsize=8, loc="upper right")
    return ax
