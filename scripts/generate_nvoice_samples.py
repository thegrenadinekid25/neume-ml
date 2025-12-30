#!/usr/bin/env python3
"""Test N-voice generation with variety of textures."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import soundfile as sf

from src.synthesis.chord_types import ChordSpec, ChordQuality
from src.synthesis.voicing import generate_n_voice_voicing, VoicingStyle
from src.synthesis.fluidsynth_renderer import FluidSynthRenderer
from src.utils.audio import load_config, resolve_soundfont_path


def main():
    """Generate N-voice test samples."""
    # Load configuration
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "synthesis.yaml"
    config = load_config(str(config_path))

    # Test cases: (chord, voice_count, style, description)
    test_cases = [
        (ChordSpec(root=2, quality=ChordQuality.MAJOR), 4, VoicingStyle.OPEN, "Dmaj_4v_open"),
        (ChordSpec(root=2, quality=ChordQuality.MAJOR), 8, VoicingStyle.OPEN, "Dmaj_8v_open"),
        (ChordSpec(root=2, quality=ChordQuality.MAJOR), 16, VoicingStyle.OPEN, "Dmaj_16v_open"),
        (ChordSpec(root=2, quality=ChordQuality.MAJOR), 8, VoicingStyle.CLUSTER, "Dmaj_8v_cluster"),
        (ChordSpec(root=9, quality=ChordQuality.MINOR), 12, VoicingStyle.OPEN, "Amin_12v_open"),
        (ChordSpec(root=7, quality=ChordQuality.DOMINANT_7), 8, VoicingStyle.CLOSE, "G7_8v_close"),
    ]

    output_dir = project_root / "data" / "output" / "nvoice_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get synthesis parameters from config
    synth_config = config["synthesis"]
    sample_rate = synth_config["sample_rate"]
    duration = synth_config["default_duration_sec"]
    velocity = synth_config["default_velocity"]

    # Resolve soundfont path
    soundfont_path = resolve_soundfont_path(config)

    print(f"Using soundfont: {soundfont_path}")
    print(f"Output directory: {output_dir}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration} sec")
    print()

    # Check if soundfont exists
    if not Path(soundfont_path).exists():
        print(f"ERROR: Soundfont not found at {soundfont_path}")
        sys.exit(1)

    # Initialize renderer
    with FluidSynthRenderer(
        soundfont_path=soundfont_path,
        sample_rate=sample_rate,
        choir_program=synth_config["soundfont"]["choir_program"],
    ) as renderer:
        for chord, n_voices, style, name in test_cases:
            # Generate voicing
            voicing = generate_n_voice_voicing(
                chord,
                voice_count=n_voices,
                style=style,
                seed=42,
            )

            print(f"{name}:")
            print(f"  Chord: {chord.name}")
            print(f"  Voices: {voicing.voice_count}")
            print(f"  Style: {voicing.style.value}")
            print(f"  Notes: {voicing.notes}")
            print(f"  Span: {voicing.span} semitones")

            # Verify constraints
            notes = voicing.to_midi_notes()
            is_valid = True
            for i in range(len(notes) - 1):
                if notes[i] >= notes[i + 1]:
                    print(f"  WARNING: Non-ascending notes at position {i}")
                    is_valid = False

            if is_valid:
                print(f"  Validation: OK (ascending, no unisons)")

            # Render audio
            audio = renderer.render_chord(
                voicing,
                duration_sec=duration,
                velocity=velocity,
            )

            # Save to file
            notes_str = "-".join(str(n) for n in voicing.notes[:4])  # First 4 notes for filename
            if len(voicing.notes) > 4:
                notes_str += f"-etc{len(voicing.notes)}"
            filename = f"{name}_{notes_str}.wav"
            filepath = output_dir / filename
            sf.write(filepath, audio, sample_rate)

            print(f"  Saved: {filename}")
            print()

    print(f"Done! Generated {len(test_cases)} test samples.")
    print(f"Listen to them in: {output_dir}")


if __name__ == "__main__":
    main()
