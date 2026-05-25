#!/usr/bin/env python3
"""Execute code cells from a notebook sequentially without nbconvert.
This is a pragmatic fallback when nbconvert/nbformat aren't available.
"""
import json
import sys
from pathlib import Path
nb_path = Path(__file__).resolve().parents[1] / 'PyANNOW' / 'notebooks' / '06_pyannow_v2_patterns.ipynb'
print('Notebook path:', nb_path)
nb = json.loads(nb_path.read_text())
# prepare execution namespace
ns = {'__name__': '__main__'}
# set working dir to notebook dir
nb_dir = nb_path.parent
print('Working dir:', nb_dir)
import os
os.chdir(str(nb_dir))
# Provide a minimal dummy matplotlib if not installed so plotting cells won't fail
try:
    import matplotlib  # noqa: F401
except Exception:
    import types
    mpl = types.ModuleType('matplotlib')
    pyplot = types.ModuleType('matplotlib.pyplot')
    pyplot.rcParams = {}
    def subplots(nrows=1, ncols=1, figsize=None):
        ax = types.SimpleNamespace(plot=lambda *a, **k: None,
                                   scatter=lambda *a, **k: None,
                                   set=lambda *a, **k: None)
        return (None, ax if nrows * ncols == 1 else [ax])
    pyplot.subplots = subplots
    pyplot.tight_layout = lambda *a, **k: None
    pyplot.show = lambda *a, **k: None
    mpl.pyplot = pyplot
    sys.modules['matplotlib'] = mpl
    sys.modules['matplotlib.pyplot'] = pyplot
# execute code cells
for i, cell in enumerate(nb['cells'], 1):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    print(f'--- Executing cell {i} ---')
    try:
        exec(compile(src, f'<cell {i}>', 'exec'), ns)
    except Exception as e:
        print(f'ERROR in cell {i}:', e)
        raise
print('All code cells executed (or error raised).')
