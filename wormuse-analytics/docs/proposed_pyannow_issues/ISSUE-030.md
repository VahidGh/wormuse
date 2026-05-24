## ISSUE-030 — Step 0 mislabelled "random / no NAML" `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P3 |
| **Severity** | Narrative — reader assumes Step 0 is random; it isn't |

**Description.** Notebook 03 cell 5 markdown:

> **Step 0 — Baseline: rule-based neuron→note mapping**
> **No learning.** Body-wave phase → pentatonic pitch. … **Result:** random-sounding, no structure matching Chopin.

And the summary table in `_build_naml_progression_nb.py`:

> | 0 | None | **Random notes** | No structure |

But Step 0 *isn't* random — it runs the deterministic body-wave forward model (`run_forward_fast` with fixed `drive_freq_hz=1.5, drive_amplitude=12.0, random_seed=42`) and produces phase-gated note events. Its IOI distribution has high regularity (single peak near one body-wave period); only the pitch/Chopin alignment is "random-like" because there's no learning, not because the timing is random.

This wording matters because the central diagnostic question of ISSUE-016 — "why does Step 0 win?" — is much easier to answer correctly when the reader knows Step 0 is *structured* (sparse, regular) rather than *random* (which would suggest a metric error of a different kind).

**Fix plan.** Re-label Step 0 in three places:

1. Notebook cell 5 markdown → "Step 0 — Deterministic body-wave baseline".
2. Summary table → "0 | None | **Deterministic body-wave** (no Chopin-aware learning) | Periodic, ~1.5 Hz rhythm".
3. Loss-progression chart x-tick → "Step 0 (body-wave)".

Add one line to the §Step 0 markdown: "*The IOI distribution of this step is the body-wave period, not a random distribution — see §L00 of `wormuse-analytics/notebooks/01_appstat_lecture_audit.ipynb` for the visualisation.*"

**Affected files.**
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — cells 5, 20 x-ticks, 24 summary table.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same.
- `PyANNOW/TODO.md` — this entry.
