# wormuse

![version](https://img.shields.io/badge/version-v2.0.0-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![python](https://img.shields.io/badge/python-3.10--3.13-blue)
![built with Claude Code](https://img.shields.io/badge/built%20with-Claude%20Code-blueviolet?logo=anthropic)

> **A *C. elegans*–driven musical simulator — and its inverse.**
>
> **v1.x (Worm → Music):** The 302-neuron nervous system, simulated by OpenWorm (Sibernetic + C302), is wired into a physical piano model. Each neuron firing triggers a hammer; muscle contractions modulate timing and dynamics. **Ion channels are the tuning fork** — a physics-informed neural network learns the kinetics that make the worm's activity musically coherent.
>
> **v2.0.0 (Music → Worm Dance):** The inverse pipeline. Chopin's Nocturne in C# minor is decomposed into K=8 recurring patterns via **RSVD + K-means** (NAML Eckart-Young), ranked by biological **Pearson excitability**, and mapped onto 96 body-wall muscles via **least-squares regression**. The worm dances to Chopin in real time in the browser — body wave, locomotion trail, neural circuit panel, and pattern timeline included. Open `PyANNOW/presentation/index_v2.html` for the full 24-slide NAML presentation.

Wormuse is a robot pianist, but with the worm as the composer. It is designed as a project integrating some of Politecnico di Milano courses:

| Course                                                     | Sub-project                                 | Focus                                                                                         |
| ---------------------------------------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **Advanced Methods for Scientific Computing (AMSC)** | [`wormuse-sim/`](./wormuse-sim/)             | C++ simulator core: OpenWorm wrapper + piano FEM + neuron→MIDI bridge                        |
| **Numerical Analysis for Machine Learning (NAML)**   | [`PyANNOW/`](./PyANNOW/)                     | 96-cell Boyle worm → Chopin mapper; 9-step NAML progression (SVD→PCA→Ridge→MLP→RF→PINN) |
| **Applied Statistics (AppStat)**                     | [`wormuse-analytics/`](./wormuse-analytics/) | Python notebooks: PCA / clustering / regression / RF / classification on worm-music data      |

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

## AI Attribution

This project was built with **[Claude Code](https://www.anthropic.com/claude-code)** (Anthropic).
Claude acted as the principal development engineer: implementing all Python modules, C++ skeletons,
tests, documentation, notebooks, and the Reveal.js presentation — under the scientific direction
and design decisions of Vahid Ghayoomie.

See **[`AI_CONTRIBUTIONS.md`](./AI_CONTRIBUTIONS.md)** for:

- Module-by-module contribution breakdown
- Where human judgment was essential (and where it corrected the AI)
- Known AI limitations encountered
- How to reproduce the workflow

## Acknowledgements

- [OpenWorm](https://openworm.org/) — Sibernetic SPH body simulator, C302 nervous system, ChannelWorm ion-channel models.
- *Time Domain Simulation of a Piano* (Chabassier et al.) — physical piano modeling reference (Parts 1 + 2).
- Prof. Miglio (NAML), Prof. Formaggia (AMSC), and Prof. Massi (AppStat).
- Vahid Ghayoomie's own channelworm and SC-PINN work informing the ion-channel architecture.
- **Claude** (Anthropic) — engineering implementation, code, tests, documentation, and design.
