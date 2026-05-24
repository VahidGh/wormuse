## ISSUE-018 — Builder script desync after ISSUE-017 fix `[@appstat-audit]`

| Field | Value |
|---|---|
| **Status** | 🔴 Open |
| **Priority** | P1 |
| **Severity** | Correctness — next rebuild silently erases the ISSUE-017 fix |

**Description.** `PyANNOW/notebooks/_build_naml_progression_nb.py` is the source of truth for `03_pyannow_naml_progression.ipynb` — running it overwrites the notebook on disk. The ISSUE-017 fix (per-step `musical_f1` + `ioi_similarity` dicts, two-panel chart in cell 20, summary table additions in cell 24) was applied directly to the executed notebook but **not back-ported into the builder script**. Comparing file mtimes confirms the desync (notebook newer than builder). The next time someone re-runs the builder, the F1 work is gone and we are back to "Step 0 wins" on a single onset_loss panel.

**Reproduction.**
```bash
grep -n 'musical_f1\|f1_scores\|ioi_similarity' PyANNOW/notebooks/_build_naml_progression_nb.py
# (no output — none of the ISSUE-016/017 names are referenced)
python3 PyANNOW/notebooks/_build_naml_progression_nb.py
# notebook is rewritten without F1 tracking
```

**Fix plan.**
1. Add to the imports cell:
   ```python
   from pyannow.targets.midi_target import (
       parse_midi, note_onsets, onset_loss, piano_roll,
       musical_f1, ioi_similarity)   # ← add these two
   ```
2. After each `losses[...] = L_xxx` line in cells 6 / 10 / 12 / 14 / 16, also append:
   ```python
   f1_x  = musical_f1(onsets_x, t_on_chopin, window_s=DURATION)
   ioi_x = ioi_similarity(onsets_x, t_on_chopin, window_s=DURATION)
   f1_scores[step_name]  = f1_x['f1']
   ioi_scores[step_name] = ioi_x
   ```
3. Initialise `f1_scores = {}` and `ioi_scores = {}` in the same cell where `losses = {}` is created (cell 6 — Step 0).
4. Replace the cell-20 single-bar-chart block with the two-panel layout currently present in the executed notebook.
5. Add a guard at the top of the builder:
   ```python
   # If the notebook on disk has been hand-edited, abort to avoid clobbering work.
   # Compare a hash of cell 20 against the version the builder is about to emit.
   ```
6. Re-run the builder, diff the produced notebook against the current one to confirm parity, then commit both.

**Affected files.**
- `PyANNOW/notebooks/_build_naml_progression_nb.py` — port lines from notebook cells 2, 6, 10, 12, 14, 16, 20, 24.
- `PyANNOW/notebooks/03_pyannow_naml_progression.ipynb` — should be byte-identical after the next rebuild.
- `PyANNOW/TODO.md` — this entry.
