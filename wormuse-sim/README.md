# wormuse-sim

**The AMSC sub-project.** C++ simulator core: OpenWorm wrapper + piano simulator + neuron→MIDI bridge.

Built in the MK Docker image (`quay.io/pjbaioni/amsc_mk:2025`), with `gcc-glibc/11.2.0 + eigen + lis + dealii` modules loaded.

---

## Layout

```
wormuse-sim/
├── CMakeLists.txt
├── src/
│   ├── ow_bridge/        Drives the OpenWorm container (Sibernetic + C302)
│   ├── piano_sim/        Physical piano: strings (Phase 2) → soundboard FEM (Phase 5)
│   ├── neuron_to_midi/   Spike events → MIDI / piano triggers
│   └── orchestrator/     Main timeline integrator
├── apps/
│   └── wormuse_sim.cpp   CLI entrypoint
├── tests/                CTest unit tests
└── cmake/                CMake helpers (find modules, etc.)
```

## Build

```bash
docker compose run --rm amsc-mk bash -c '
  source /u/sw/etc/profile.d/mk.sh
  module load gcc-glibc/11.2.0 eigen lis dealii
  cd /work/wormuse-sim
  mkdir -p build && cd build
  cmake .. && make -j
'
```

## Lecture map (AMSC + cross-course)

This sub-project exercises lectures from AMSC, NMPDE, NLA, and PC. Each row points at the file where the concept lives once implemented.

### Modern C++ (AMSC L03–L08)

| Lecture | Concept | File |
|---|---|---|
| L03 IntroToCpp | `std::vector`, references, range-for | throughout |
| L04 Functions | overloading, default args, `[[nodiscard]]` | `src/piano_sim/string_1d.hpp` (planned) |
| L05 SmartPointers | `unique_ptr<WormSimulator>`, RAII for the OpenWorm subprocess | `src/ow_bridge/docker_runner.hpp` |
| L06 Classes | virtual base `PianoString`, override, `= default` dtors | `src/piano_sim/string.hpp` |
| L07 ClassTemplates | `PianoString<N_modes>` (mode count at compile time) | `src/piano_sim/modal_string.hpp` |
| L08 StandardLibrary | `std::variant<MidiNoteOn, MidiNoteOff>`, `std::optional<NextSpike>` | `src/neuron_to_midi/events.hpp` |

### Build systems & tooling (AMSC L09)

| Lecture | Concept | File |
|---|---|---|
| L09 Static/Shared libs | `wormuse_core` as a STATIC library; apps link against it | `CMakeLists.txt` |
| Notes — Makefile/CMake | Two-flavor build: top-level CMake + module-aware Make | `cmake/find_dealii.cmake` |
| Notes — gdb | Provide `make debug` target with `-g -O0 -fsanitize=address` | `cmake/sanitizers.cmake` |

### Parallel computing (AMSC L10–L12 + PC course)

| Lecture | Concept | File |
|---|---|---|
| AMSC L11 (MPI) | Optional MPI for multi-node piano FEM | `src/piano_sim/dist_assembler.hpp` (Phase 5+) |
| AMSC L12 (OpenMP) | `#pragma omp parallel for` over strings; reduction over modes | `src/piano_sim/multi_string.cpp` |
| PC L8-10 (patterns) | The piano timeline is a **map**; reductions for energy norm; **stencil** for FDM string | `src/piano_sim/string_1d.cpp` |
| PC L5 (GPU) | (Stretch) CUDA backend for the soundboard | `src/piano_sim/cuda/` |

### Eigen + NLA (cross-course)

| Lecture | Concept | File |
|---|---|---|
| AMSC L08 + NLA full course | `Eigen::SparseMatrix`, `ConjugateGradient` for stiffness solve | `src/piano_sim/soundboard_fem.cpp` |
| NLA L (iterative + LIS) | LIS-backed iterative solver as alternative backend | `src/piano_sim/lis_backend.cpp` |

### Numerical methods for PDEs (NMPDE)

| Lecture | Concept | File |
|---|---|---|
| Heat-style derivation | Soundboard wave equation, weak form, FEM assembly | `docs/math_derivations/soundboard_weak_form.md` |
| Lab 4 (heat / parabolic) | θ-method for time integration | `src/piano_sim/time_integrator.hpp` |
| Lab 7 (Stokes) | (Stretch) air radiation coupling | future |

The piano FEM in Phase 5 will **reuse the CMake + CI infrastructure of the user's `nmpde-projects/Heat.cpp`** (manufactured solution tests, verification workflow).

---

## Phases

- **Phase 1** — `ow_bridge` minimal driver: spawn container, read spikes.
- **Phase 2** — `piano_sim::String1D` (1D wave eq, OpenMP per-string).
- **Phase 5** — `piano_sim::SoundboardFEM` (deal.II, Q1 elements, CG solver).
- **Phase 6** — `orchestrator` end-to-end: worm → spikes → MIDI → audio WAV.

See [../ROADMAP.md](../ROADMAP.md) for the full plan.

## Conventions

- C++17 by default (C++20 only where it materially improves clarity).
- Compile with `-Wall -Wextra -Wpedantic`. Treat warnings as errors in CI.
- Eigen includes via `$mkEigenInc` env var (set by the MK eigen module).
- deal.II via `find_package(deal.II 9.5 REQUIRED HINTS $ENV{mkDealiiPrefix})`.
- For targets using `deal_ii_setup_target`, use the **plain** form of `target_link_libraries` (no `PRIVATE`/`PUBLIC` keywords).

## When stuck

- `polimi-amsc` skill has full references: `cheatsheet-cpp.md`, `cheatsheet-parallel.md`, `eigen-patterns.md`, `build-tools.md`.
- `polimi-nmpde` skill has `dealii-patterns.md` + a working Poisson skeleton.
- `polimi-nla` skill for solver choice (CG vs GMRES vs LIS).
- `polimi-pc` skill for parallel patterns and OpenMP/CUDA depth.
