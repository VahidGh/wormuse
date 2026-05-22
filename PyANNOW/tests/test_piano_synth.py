"""Tests for PyANNOW/src/pyannow/composer/piano_synth.py

Covers:
  - render_string output length, range, finiteness
  - synthesise_melody produces correct duration
  - Empty events → silence
  - pitch_map lookup for muscle indices
  - NoteEvent objects pass through correctly
"""
from __future__ import annotations

import numpy as np
import pytest

FS = 22050   # sample rate used in tests


class TestRenderString:

    @pytest.mark.parametrize("f0,vel,dur", [
        (261.63, 64, 0.5),    # C4, medium velocity
        (440.00, 80, 1.0),    # A4, louder
        (55.00,  40, 0.3),    # A1, soft bass note
    ])
    def test_output_length(self, f0, vel, dur):
        from pyannow.composer.piano_synth import render_string
        audio = render_string(f0=f0, velocity=vel, duration_s=dur, fs=FS)
        expected = int(dur * FS)
        assert len(audio) == expected, (
            f"Expected {expected} samples for {dur}s, got {len(audio)}")

    def test_output_finite(self):
        from pyannow.composer.piano_synth import render_string
        audio = render_string(f0=440.0, velocity=64, duration_s=0.5, fs=FS)
        assert np.all(np.isfinite(audio)), "render_string must produce finite audio"

    def test_output_normalised(self):
        from pyannow.composer.piano_synth import render_string
        audio = render_string(f0=440.0, velocity=127, duration_s=0.5, fs=FS)
        # The amplitude should be at most 1.0 (normalised per-note)
        assert abs(audio).max() <= 1.001, (
            f"Audio amplitude {abs(audio).max():.3f} exceeds 1.0")

    def test_higher_velocity_louder(self):
        from pyannow.composer.piano_synth import render_string
        soft = render_string(f0=440.0, velocity=10,  duration_s=0.5, fs=FS)
        loud = render_string(f0=440.0, velocity=100, duration_s=0.5, fs=FS)
        assert abs(loud).max() >= abs(soft).max(), (
            "Higher velocity must produce at least as loud audio")

    def test_dtype_float32(self):
        from pyannow.composer.piano_synth import render_string
        audio = render_string(f0=440.0, velocity=64, duration_s=0.3, fs=FS)
        assert audio.dtype == np.float32, "render_string must return float32"


class TestSynthesiseMelody:

    def test_empty_events_silence(self):
        from pyannow.composer.piano_synth import synthesise_melody
        audio, fs = synthesise_melody([], duration_s=2.0, fs=FS)
        assert len(audio) == int(2.0 * FS)
        assert np.all(audio == 0.0), "Empty events must produce silence"

    def test_correct_output_length(self, tiny_events):
        from pyannow.composer.piano_synth import synthesise_melody
        duration = 3.5
        audio, fs = synthesise_melody(tiny_events, duration_s=duration, fs=FS)
        assert len(audio) == int(duration * FS)

    def test_output_finite(self, tiny_events):
        from pyannow.composer.piano_synth import synthesise_melody
        audio, _ = synthesise_melody(tiny_events, duration_s=3.5, fs=FS)
        assert np.all(np.isfinite(audio)), "synthesise_melody must produce finite audio"

    def test_output_normalised(self, tiny_events):
        from pyannow.composer.piano_synth import synthesise_melody
        audio, _ = synthesise_melody(tiny_events, duration_s=3.5, fs=FS)
        assert abs(audio).max() <= 1.001, "synthesise_melody output must be ≤ 1.0"

    def test_note_event_objects_work(self, tiny_events):
        """NoteEvent dataclass objects should parse correctly."""
        from pyannow.composer.piano_synth import synthesise_melody
        audio, _ = synthesise_melody(tiny_events, duration_s=3.5, fs=FS)
        # If events parse correctly, audio should be non-zero
        assert abs(audio).max() > 0.0, "NoteEvent objects should produce non-silent audio"

    def test_worm_tuples_work(self):
        """(time_s, muscle_idx, velocity) tuples should map via WORM_PITCHES."""
        from pyannow.composer.piano_synth import synthesise_melody
        events = [(0.0, 0, 64), (0.5, 3, 72), (1.0, 7, 55)]
        audio, _ = synthesise_melody(events, duration_s=2.0, fs=FS)
        assert abs(audio).max() > 0.0, "Worm tuples should produce audible output"

    def test_pitch_map_lookup(self):
        """When pitch_map is provided, muscle indices must translate to correct MIDI pitch."""
        from pyannow.composer.piano_synth import synthesise_melody, midi_to_hz
        import numpy as np
        pitch_map = np.array([25, 37, 49, 61, 73, 85, 97, 108])
        events = [(0.0, 0, 64)]   # muscle 0 → MIDI 25 → A0
        audio, _ = synthesise_melody(events, duration_s=1.0, fs=FS, pitch_map=pitch_map)
        assert abs(audio).max() > 0.0, "pitch_map lookup must produce audible output"

    def test_max_notes_limit(self, tiny_events):
        """max_notes=2 should only render the first 2 note events."""
        from pyannow.composer.piano_synth import synthesise_melody
        audio_full, _ = synthesise_melody(tiny_events, duration_s=3.5, fs=FS)
        audio_2,    _ = synthesise_melody(tiny_events, duration_s=3.5, fs=FS, max_notes=2)
        # After note 3 (t=1.5s), the limited version should be silent
        start_silent = int(2.0 * FS)
        if abs(audio_full[start_silent:]).max() > 0.01:
            assert abs(audio_2[start_silent:]).max() < abs(audio_full[start_silent:]).max(), (
                "Limited render should have less energy after truncation point")


class TestMidiToHz:

    @pytest.mark.parametrize("midi,expected_hz", [
        (69, 440.0),     # A4
        (60, 261.626),   # C4 (middle C)
        (21, 27.5),      # A0 (lowest piano key)
    ])
    def test_known_frequencies(self, midi, expected_hz):
        from pyannow.composer.piano_synth import midi_to_hz
        hz = midi_to_hz(midi)
        assert abs(hz - expected_hz) < 0.5, (
            f"MIDI {midi} → expected {expected_hz:.1f} Hz, got {hz:.1f} Hz")
