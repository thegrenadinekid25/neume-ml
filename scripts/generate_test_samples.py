#!/usr/bin/env python3
"""Generate test samples to validate the synthesis pipeline."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import soundfile as sf

from src.synthesis.chord_types import ChordSpec, ChordQuality
from src.synthesis.voicing import generate_voicing
from src.synthesis.fluidsynth_renderer import FluidSynthRenderer
from src.utils.audio import load_config, resolve_soundfont_path


def main():
    """Generate test chord samples."""
    # Load configuration
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "synthesis.yaml"
    config = load_config(str(config_path))

    # Test chords: D major, A minor, G7 (the opening of many pieces)
    test_chords = [
        ChordSpec(root=2, quality=ChordQuality.MAJOR),     # D major
        ChordSpec(root=9, quality=ChordQuality.MINOR),     # A minor
        ChordSpec(root=7, quality=ChordQuality.DOMINANT_7),  # G7
    ]

    output_dir = project_root / "data" / "output" / "test"
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
        print()
        print("Please download FluidR3_GM.sf2 from:")
        print("  https://member.keymusician.com/Member/FluidR3_GM/index.html")
        print()
        print("And place it in data/soundfonts/FluidR3_GM.sf2")
        sys.exit(1)

    # Initialize renderer
    with FluidSynthRenderer(
        soundfont_path=soundfont_path,
        sample_rate=sample_rate,
        choir_program=synth_config["soundfont"]["choir_program"],
    ) as renderer:
        for chord in test_chords:
            # Generate voicing (use chord root as seed for reproducibility)
            voicing = generate_voicing(chord, seed=chord.root * 1000)

            print(f"Chord: {chord.name}")
            print(f"  Pitch classes: {chord.get_pitch_classes()}")
            print(f"  Voicing: {voicing}")

            # Render audio
            audio = renderer.render_chord(
                voicing,
                duration_sec=duration,
                velocity=velocity,
            )

            # Save to file
            filename = f"{chord.name}_{voicing.bass}-{voicing.tenor}-{voicing.alto}-{voicing.soprano}.wav"
            filepath = output_dir / filename
            sf.write(filepath, audio, sample_rate)

            print(f"  Saved: {filename}")
            print()

    print("Done! Generated 3 test samples.")
    print(f"Listen to them in: {output_dir}")


if __name__ == "__main__":
    main()
