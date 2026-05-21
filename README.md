# wormuse

> **A *C. elegans*–driven musical simulator.**
> The 302-neuron nervous system, simulated by OpenWorm (Sibernetic + C302), is wired into a physical piano model. Each neuron firing triggers a hammer; muscle contractions modulate timing and dynamics. **Ion channels are the tuning fork** — a physics-informed neural network learns the kinetics that make the worm's activity musically coherent. Move the worm → hear the melody.

Wormuse is the spiritual successor to [polimuse](https://github.com/.../polimuse), but with the worm as the composer rather than a robot pianist. It is designed as a thesis-grade integration of three Politecnico di Milano courses:

| Course | Sub-project | Focus |
|---|---|---|
| **Advanced Methods for Scientific Computing (AMSC)** | [`wormuse-sim/`](./wormuse-sim/) | C++ simulator core: OpenWorm wrapper + piano FEM + neuron→MIDI bridge |
| **Numerical Analysis for Machine Learning (NAML)** | [`PyANNOW/`](./PyANNOW/) | JAX/Flax/Optax composer **conditioned on a PINN-tuned ion channel model** |
| **Applied Statistics (AppStat)** | [`wormuse-analytics/`](./wormuse-analytics/) | Python notebooks: PCA / clustering / regression / classification on the worm-music data |

Plus cross-cutting use of Numerical Linear Algebra (NLA), Numerical Methods for PDEs (NMPDE), Parallel Computing (PC), and Software Engineering for HPC (SE4HPC).

## Quick start

```bash
git clone <local path>/wormuse
cd wormuse
docker compose up amsc-mk openworm        # pulls the user's existing images
# explore the three sub-projects:
#   wormuse-sim/        — C++ simulator (build in MK docker)
#   PyANNOW/            — Python composer (local Python env with JAX/Flax/Optax)
#   wormuse-analytics/  — Python notebooks (numpy / sklearn / statsmodels)
```

## Architecture in one diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          wormuse                                          │
│                                                                          │
│   ┌─ wormuse-sim (C++) ────────┐    ┌─ PyANNOW (Python/JAX) ─────────┐  │
│   │  • OpenWorm Sibernetic     │ →  │  • SC-PINN: ion channels       │  │
│   │  • C302 nervous system     │    │  • Neural-state encoder        │  │
│   │  • Piano FEM (deal.II)     │ ←  │  • Composer (latent → MIDI)    │  │
│   │  • Neuron→MIDI bridge      │    └────────────────────────────────┘  │
│   └────────────────────────────┘                                         │
│              ↓                                                           │
│   ┌─ shared/ (data formats) ───────────────────────────────────────┐    │
│   │  WCON pose · spike events · MIDI · piano-state JSON            │    │
│   └────────────────────────────────────────────────────────────────┘    │
│              ↓                                                           │
│   ┌─ wormuse-analytics (Python notebooks) ─────────────────────────┐    │
│   │  PCA · clustering · regression · classification                │    │
│   │  Music-quality scoring · feedback into PyANNOW training        │    │
│   └────────────────────────────────────────────────────────────────┘    │
│              ↓                                                           │
│   ┌─ ui/ ──────────────────────────────────────────────────────────┐    │
│   │  Static HTML + Three.js (GitHub Pages)                         │    │
│   │  JupyterLite (Pyodide) live exploration                        │    │
│   └────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full description and [ROADMAP.md](./ROADMAP.md) for the phased plan.

## Ion channels — the centerpiece

The Hodgkin-Huxley gating variables `(m_∞, h_∞, τ_m, τ_h)` of the worm's neurons are learned by a **Physics-Informed Neural Network**. The user's existing ion-channel PINN research (channelworm + SC-PINN) is the academic anchor. The UI exposes these parameters as live controls: change a single ion-channel parameter → the music morphs in tempo and timbre. See [ION_CHANNELS.md](./ION_CHANNELS.md) for the design and [docs/SCIENTIFIC_FOUNDATION.md](./docs/SCIENTIFIC_FOUNDATION.md) for the full mathematics on both sides of the pipeline.

## Repository layout

```
wormuse/
├── README.md                ← this file
├── ARCHITECTURE.md          ← module breakdown, data flow
├── ROADMAP.md               ← 8-phase implementation plan
├── ION_CHANNELS.md          ← the PINN centerpiece (design view)
├── docs/SCIENTIFIC_FOUNDATION.md  ← biology↔piano physics derivation + references
├── LICENSE                  ← MIT
├── docker-compose.yml       ← AMSC MK + OpenWorm + JupyterLab
├── .github/workflows/       ← verification + scalability CI
│
├── wormuse-sim/             ← AMSC project (C++)
├── PyANNOW/                 ← NAML project (Python/JAX)
├── wormuse-analytics/       ← AppStat project (Python)
│
├── shared/                  ← cross-project data formats and examples
├── ui/                      ← GitHub-Pages-friendly visualization
├── docs/                    ← lecture mapping, design notes, derivations
└── scripts/                 ← convenience scripts (build, render, etc.)
```

## Course → lecture mapping

Each piece of code is tied to a specific lecture concept. See [docs/lectures/lecture-map.md](./docs/lectures/lecture-map.md) for the full table.

## License

[MIT](./LICENSE) — compatible with OpenWorm and the rest of the ecosystem.

## Acknowledgements

- [OpenWorm](https://openworm.org/) — Sibernetic SPH body simulator, C302 nervous system, ChannelWorm ion-channel models.
- *Time Domain Simulation of a Piano* (Chabassier et al.) — physical piano modeling reference (Parts 1 + 2).
- Prof. Luca Formaggia (AMSC), Profs. Beraha & Andre (AppStat 2026), and the NAML faculty (PINN lecture).
- The user's own [channelworm](#) and SC-PINN preliminary work informing the ion-channel architecture.
