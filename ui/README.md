# ui/

The interactive demo — designed to be deployed on **GitHub Pages**.

## Two-layer approach

### 1. `static/` — pure HTML + JS (main GH Pages target)

- `index.html` — single-page app
- `js/worm-viz.js` — Three.js worm body + neuron-firing animation
- `js/piano-viz.js` — piano keys + Web Audio API playback + waveform visualization
- `data/` — pre-rendered scenarios (JSON + MIDI). Generated offline by `render/`.
- `css/style.css` — minimal styling

No server, no Python in the browser. Everything is a static asset.

### 2. `notebook/wormuse.ipynb` — JupyterLite

- Runs Python in the browser via Pyodide
- Lets the user tweak a small surrogate of the PINN parameters and see the music morph
- Embeds the Three.js viz as an iframe pointing at `static/`

Hosted via [JupyterLite](https://jupyterlite.readthedocs.io/) (also a static site).

### 3. `render/` — offline rendering

Python scripts that **produce** the static `data/` folder by running the full pipeline (Sibernetic + PyANNOW + piano_sim) and saving JSONs + MIDIs.

- `render/build_static_dataset.py` — picks ≥ 10 ion-channel parameter sets, renders each, packs into `static/data/`.

This runs on a developer machine (with the MK + OpenWorm docker containers); the static site itself ships only the outputs.

## GitHub Pages deploy

`.github/workflows/gh-pages.yml` (Phase 7) will:

1. Restore `static/data/` from a release artifact (since rendering is expensive).
2. Build JupyterLite via `jupyter-lite build` against `notebook/`.
3. Combine `static/` + JupyterLite build → publish to `gh-pages` branch.

The URL will be `https://vahidghayoomie.github.io/wormuse/` (when made public).

## Tech choices

- **Three.js** for the 3D worm rendering (mature, runs in any modern browser).
- **Web Audio API** (native, no library) for playing the synthesized piano.
- **Tone.js** (optional) if we want MIDI playback shortcuts.
- **JupyterLite** for live exploration without server.
- **No frameworks** — vanilla JS module pattern. Keeps the site lightweight.

## Phases

Most of `ui/` is empty until Phase 7. Phase 0 just stakes out the layout.
