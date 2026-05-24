## ISSUE-019 — Cell 20 left panel still misleads readers `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P1 |
| **Severity** | UX / interpretation — readers still ask "why does Step 0 win?" |

**Description.** Cell 20 of `03_pyannow_naml_progression.ipynb` shows two side-by-side bar charts: onset_loss on the **left** (lower = better, but gameable) and musical_f1 on the **right**. The eye reads left-to-right and lands on Step 0 = 0.0022 first; the "(gameable — low ≠ musical)" annotation is too subtle to overcome that first impression. End-users (including the author asking the diagnostic question for this audit) read the left panel and conclude Step 0 is best, even though ISSUE-016/017 explicitly say it isn't. The fix is purely cosmetic but high-leverage.

**Fix plan.**

1. **Swap the panel order** so F1 (the trustworthy metric) is on the LEFT (read first):
   ```python
   fig, axes = plt.subplots(1, 2, figsize=(13, 4))
   # ── Left: musical_f1 (higher = better) ─────────────────
   #   (was on the right)
   # ── Right: onset_loss (gameable — diagnostic only) ────
   #   (was on the left)
   ```
2. **Visually demote the onset_loss panel**: tint its bars grey, add a red banner title `DIAGNOSTIC ONLY — see ISSUE-016`, and put the y-axis label `onset_loss (↓, gameable)` in red.
3. **Optionally** add a third panel `1 − F1` so all three charts share the convention "lower = worse" (or invert the F1 panel to "higher = better" — pick one).
4. Update the markdown summary in cell 24 to lead with the F1 column.

**Snippet for the new layout:**
```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
# Left — F1 (read first)
vals_f1 = [f1_scores[k] for k in f1_scores]
axes[0].bar(range(len(vals_f1)), vals_f1, color='#2ca02c', edgecolor='white', width=0.7)
axes[0].set(ylabel='musical F1 (↑)', title='Musical F1 per step',
            xticks=range(len(f1_scores)), xticklabels=list(f1_scores))
axes[0].tick_params(axis='x', rotation=15, labelsize=8)

# Right — onset_loss (diagnostic only, greyed out)
vals_ol = [losses[k] for k in losses if 'PINN' not in k]
bars = axes[1].bar(range(len(vals_ol)), vals_ol, color='#888', alpha=0.55,
                   edgecolor='#222', width=0.7)
axes[1].set(ylabel='onset_loss (gameable)', title='⚠ DIAGNOSTIC ONLY — see ISSUE-016')
axes[1].title.set_color('#c00')
```

**Affected files.**
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — cell 20.
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — same change (after ISSUE-018 lands).
- `PyANNOW/TODO.md` — this entry.
