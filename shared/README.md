# shared/

Bridge between the three sub-projects. Contracts and reference data live here.

## Contents

```
shared/
├── data_formats/         JSON / WCON / MIDI schemas + version bumps
├── parameter_schemas/    Pydantic models (Python) + matching C++ structs
└── examples/             Pre-computed reference scenarios for tests & UI
```

### `data_formats/`

The C++ simulator writes and the Python projects read. Schemas are versioned: `spike_event.v1.json`, `pose.v1.wcon`, etc.

- `spike_event.json` — `[{"t": float seconds, "neuron_id": int, "voltage_mV": float, "source": "C302"}]`
- `pose.wcon` — OpenWorm's standard WCON format (Worm Compute Object Notation)
- `piano_state.json` — `{"strings": [{"id": int, "displacement": [...], "velocity": [...]}, ...], "soundboard": {...}}`
- `midi.mid` — standard MIDI; no custom format

### `parameter_schemas/`

Tiny Pydantic models defining the parameter spaces (so PyANNOW + wormuse-analytics agree on names + units):

- `IonChannelParams` — `(V_thresh, tau_m, tau_h, g_K, g_Na, ...)` with units in mV / ms / mS·cm⁻²
- `PianoConfig` — `(num_strings, fundamental_freq, soundboard_size, ...)`
- `WormConfig` — `(duration_seconds, openworm_image_tag, c302_subset, ...)`

When the C++ side needs the same struct, mirror it in a `.hpp` next to the Python class with matching field names.

### `examples/`

Versioned, small reference scenarios for tests and the UI demo. Naming: `dataset_v{N}/run_{NNN}/`.

```
examples/
├── dataset_v0/
│   └── readme.md          (Phase 0 placeholder)
├── dataset_v1/             (Phase 4 will populate this with ≥ 50 scenarios)
└── dataset_v2/             (Phase 6 will populate with end-to-end music)
```

Large datasets are `.gitignore`d; only the smallest reference scenarios committed.
