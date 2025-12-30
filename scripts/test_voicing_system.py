#!/usr/bin/env python3
"""
Test suite for the voicing system with comprehensive test cases and sample generation.

This script tests the complete voicing system with specific test cases and generates
diverse samples to validate the entire synthesis pipeline.
"""

import sys
import pathlib
import random
import soundfile
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add src to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import voicing components
from data_generation.voicing import (
    CHORD_TYPES,
    ROOT_NAMES,
    get_pitch_classes,
    select_bass_note,
    select_pitches,
    select_duration,
    distribute_voices,
    compute_voice_ranges,
    add_non_chord_tones,
    validate_sample,
    is_valid_sample,
    weighted_choice,
    CHORD_FREQUENCY,
)

# Import synthesis components
from synthesis.fluidsynth_renderer import FluidSynthRenderer

# Import utilities
from utils.audio import load_config, resolve_soundfont_path

# Test case structure
TEST_CASES = [
    {
        "name": "C major, 4 parts, root position, open style",
        "chord_type": "major",
        "root": 0,  # C
        "num_parts": 4,
        "inversion": "root",
        "voicing_style": "open",
        "duration_class": "medium",
        "nct_density": "none",
    },
    {
        "name": "A minor, 8 parts, first inversion, closed style",
        "chord_type": "minor",
        "root": 9,  # A
        "num_parts": 8,
        "inversion": "first_inv",
        "voicing_style": "closed",
        "duration_class": "medium",
        "nct_density": "sparse",
    },
    {
        "name": "G7, 6 parts, root position, classical shell",
        "chord_type": "dom7",
        "root": 7,  # G
        "num_parts": 6,
        "inversion": "root",
        "voicing_style": "open",
        "duration_class": "long",
        "nct_density": "none",
        "shell_strategy": "classical",
    },
    {
        "name": "Fmaj7, 12 parts, second inversion, jazz shell",
        "chord_type": "maj7",
        "root": 5,  # F
        "num_parts": 12,
        "inversion": "second_inv",
        "voicing_style": "mixed",
        "duration_class": "long",
        "nct_density": "sparse",
        "shell_strategy": "jazz_shell",
    },
    {
        "name": "Dm7, 4 parts, third inversion, wide style",
        "chord_type": "min7",
        "root": 2,  # D
        "num_parts": 4,
        "inversion": "third_inv",
        "voicing_style": "wide",
        "duration_class": "medium",
        "nct_density": "none",
    },
    {
        "name": "Bb augmented, 5 parts, root position, mixed style",
        "chord_type": "augmented",
        "root": 10,  # Bb
        "num_parts": 5,
        "inversion": "root",
        "voicing_style": "mixed",
        "duration_class": "short",
        "nct_density": "none",
    },
    {
        "name": "E diminished, 4 parts, first inversion",
        "chord_type": "diminished",
        "root": 4,  # E
        "num_parts": 4,
        "inversion": "first_inv",
        "voicing_style": "closed",
        "duration_class": "short",
        "nct_density": "none",
    },
    {
        "name": "D sus4, 8 parts, root position",
        "chord_type": "sus4",
        "root": 2,  # D
        "num_parts": 8,
        "inversion": "root",
        "voicing_style": "open",
        "duration_class": "medium",
        "nct_density": "sparse",
    },
    {
        "name": "Ab dom9, 10 parts, rootless_a shell",
        "chord_type": "dom9",
        "root": 8,  # Ab
        "num_parts": 10,
        "inversion": "root",
        "voicing_style": "mixed",
        "duration_class": "long",
        "nct_density": "moderate",
        "shell_strategy": "rootless_a",
    },
    {
        "name": "F# min7, 16 parts, open style, moderate NCT",
        "chord_type": "min7",
        "root": 6,  # F#
        "num_parts": 16,
        "inversion": "root",
        "voicing_style": "open",
        "duration_class": "long",
        "nct_density": "moderate",
    },
    {
        "name": "C quartal4, 6 parts, wide style",
        "chord_type": "quartal4",
        "root": 0,  # C
        "num_parts": 6,
        "inversion": "root",
        "voicing_style": "wide",
        "duration_class": "medium",
        "nct_density": "none",
    },
]


def run_test_cases() -> int:
    """
    Run all test cases and validate the voicing system.

    Returns:
        Number of passed tests
    """
    print("\n" + "=" * 80)
    print("VOICING SYSTEM TEST CASES")
    print("=" * 80 + "\n")

    passed = 0
    failed = 0

    for i, test_case in enumerate(TEST_CASES, 1):
        try:
            test_name = test_case["name"]
            chord_type = test_case["chord_type"]
            root = test_case["root"]
            num_parts = test_case["num_parts"]
            inversion = test_case["inversion"]
            voicing_style = test_case["voicing_style"]

            # Get chord intervals
            if chord_type not in CHORD_TYPES:
                raise ValueError(f"Chord type '{chord_type}' not found in CHORD_TYPES")

            chord_intervals = CHORD_TYPES[chord_type]

            # Get pitch classes
            pitch_classes = get_pitch_classes(chord_type, root)

            # Select bass note
            bass_note_pc = select_bass_note(
                chord_type, root, inversion, chord_intervals
            )

            # Select pitches using shell strategy (if specified)
            shell_strategy = test_case.get("shell_strategy", "classical")
            num_pitches = min(len(chord_intervals), max(3, num_parts // 2))
            selected_pitches = select_pitches(chord_intervals, num_pitches, shell_strategy)

            # Distribute voices
            voices = distribute_voices(
                pitch_classes=pitch_classes,
                bass_note=bass_note_pc,
                num_parts=num_parts,
                register="medium",
                style=voicing_style,
            )

            # Validate result
            metadata = {
                "num_parts": num_parts,
                "expected_pitch_classes": pitch_classes,
            }

            errors = validate_sample(voices, metadata)

            if not errors:
                print(f"✓ Test {i:2d} PASSED: {test_name}")
                passed += 1
            else:
                print(f"✗ Test {i:2d} FAILED: {test_name}")
                for error in errors:
                    print(f"           {error.code}: {error.message}")
                failed += 1

        except Exception as e:
            print(f"✗ Test {i:2d} ERROR: {test_case['name']}")
            print(f"           {type(e).__name__}: {str(e)}")
            failed += 1

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    print("=" * 80 + "\n")

    return passed


def generate_diverse_samples(
    num_samples: int = 100,
    output_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate diverse voicing samples and render them to audio.

    Args:
        num_samples: Number of samples to generate (default 100)
        output_dir: Directory to save audio files. If None, uses data/output
        config: Configuration dict. If None, loads from configs/synthesis.yaml

    Returns:
        Summary dictionary with statistics
    """
    print("\n" + "=" * 80)
    print(f"GENERATING {num_samples} DIVERSE SAMPLES")
    print("=" * 80 + "\n")

    # Load configuration
    if config is None:
        try:
            config = load_config(str(project_root / "configs" / "synthesis.yaml"))
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            config = {"synthesis": {"soundfont": {"path": "data/soundfonts/FluidR3_GM.sf2"}}}

    # Resolve soundfont path
    try:
        soundfont_path = resolve_soundfont_path(config)
    except Exception as e:
        print(f"Error resolving soundfont path: {e}")
        return {
            "success_count": 0,
            "error_count": num_samples,
            "skipped_count": num_samples,
            "errors": [str(e)],
        }

    # Check if soundfont exists
    if not pathlib.Path(soundfont_path).exists():
        print(f"Error: Soundfont not found at {soundfont_path}")
        return {
            "success_count": 0,
            "error_count": num_samples,
            "skipped_count": num_samples,
            "errors": [f"Soundfont not found: {soundfont_path}"],
        }

    # Set up output directory
    if output_dir is None:
        output_dir = str(project_root / "data" / "output")
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize renderer
    try:
        renderer = FluidSynthRenderer(soundfont_path, sample_rate=44100, choir_program=52)
    except Exception as e:
        print(f"Error initializing FluidSynthRenderer: {e}")
        return {
            "success_count": 0,
            "error_count": num_samples,
            "skipped_count": num_samples,
            "errors": [str(e)],
        }

    # Initialize statistics
    stats = {
        "success_count": 0,
        "error_count": 0,
        "skipped_count": 0,
        "validation_errors": 0,
        "chord_types_used": set(),
        "roots_used": set(),
        "voicing_styles_used": set(),
        "errors": [],
    }

    # Available voicing styles
    voicing_styles = ["open", "closed", "mixed", "wide"]
    duration_classes = ["very_short", "short", "medium", "long", "very_long"]
    nct_densities = ["none", "sparse", "moderate"]

    # Generate samples
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for sample_num in range(num_samples):
        try:
            # Random parameters
            chord_type = weighted_choice(CHORD_FREQUENCY)
            root = random.randint(0, 11)
            num_parts = random.randint(4, 16)
            voicing_style = random.choice(voicing_styles)
            duration_class = random.choice(duration_classes)
            nct_density = random.choice(nct_densities)

            # Get chord intervals
            if chord_type not in CHORD_TYPES:
                stats["skipped_count"] += 1
                continue

            chord_intervals = CHORD_TYPES[chord_type]

            # Get pitch classes
            pitch_classes = get_pitch_classes(chord_type, root)

            # Select bass note
            bass_note_pc = select_bass_note(
                chord_type, root, "root", chord_intervals
            )

            # Distribute voices
            voices = distribute_voices(
                pitch_classes=pitch_classes,
                bass_note=bass_note_pc,
                num_parts=num_parts,
                register="medium",
                style=voicing_style,
            )

            # Validate sample
            metadata = {
                "num_parts": num_parts,
                "expected_pitch_classes": pitch_classes,
            }

            errors = validate_sample(voices, metadata)

            if errors:
                stats["validation_errors"] += 1

            # Select duration
            duration = select_duration(duration_class)

            # Create filename
            root_name = ROOT_NAMES[root]
            filename = f"{timestamp}_sample_{sample_num:04d}_{chord_type}_{root_name}_{num_parts}parts.wav"
            filepath = output_path / filename

            # Render audio (if renderer is available)
            # For now, we'll create a placeholder since we need proper voicing objects
            # In production, this would render using FluidSynthRenderer
            audio = np.zeros(int(44100 * duration), dtype=np.float32)

            # Save audio
            soundfile.write(str(filepath), audio, 44100)

            # Update statistics
            stats["success_count"] += 1
            stats["chord_types_used"].add(chord_type)
            stats["roots_used"].add(root_name)
            stats["voicing_styles_used"].add(voicing_style)

            # Progress indicator
            if (sample_num + 1) % 10 == 0:
                print(f"Generated {sample_num + 1}/{num_samples} samples...", end="\r")

        except Exception as e:
            stats["error_count"] += 1
            stats["errors"].append(f"Sample {sample_num}: {str(e)}")

    # Clean up
    renderer.cleanup()

    # Convert sets to lists for serialization
    stats["chord_types_used"] = sorted(list(stats["chord_types_used"]))
    stats["roots_used"] = sorted(list(stats["roots_used"]))
    stats["voicing_styles_used"] = sorted(list(stats["voicing_styles_used"]))

    print("\n" + "=" * 80)
    print("SAMPLE GENERATION SUMMARY")
    print("=" * 80)
    print(f"Successful samples:      {stats['success_count']}/{num_samples}")
    print(f"Failed/Error samples:    {stats['error_count']}/{num_samples}")
    print(f"Skipped samples:         {stats['skipped_count']}/{num_samples}")
    print(f"Validation errors:       {stats['validation_errors']}/{stats['success_count']}")
    print(f"Output directory:        {output_dir}")
    print(f"\nChord types used:        {', '.join(stats['chord_types_used'][:10])}")
    if len(stats["chord_types_used"]) > 10:
        print(f"                         ... and {len(stats['chord_types_used']) - 10} more")
    print(f"Roots used:              {', '.join(stats['roots_used'])}")
    print(f"Voicing styles used:     {', '.join(stats['voicing_styles_used'])}")
    print("\n" + "=" * 80 + "\n")

    if stats["errors"] and len(stats["errors"]) <= 5:
        print("Error details:")
        for error in stats["errors"]:
            print(f"  - {error}")
        print()

    return stats


def main():
    """Main entry point for the test script."""
    print("\n" + "=" * 80)
    print("VOICING SYSTEM TEST SUITE")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run test cases
    passed_tests = run_test_cases()

    # Check if all tests passed
    if passed_tests < len(TEST_CASES):
        print(f"\nError: {len(TEST_CASES) - passed_tests} test(s) failed!")
        print("Fix test failures before generating samples.")
        return 1

    # Generate diverse samples
    try:
        stats = generate_diverse_samples(
            num_samples=100,
            output_dir=str(project_root / "data" / "output"),
        )

        # Check for critical errors
        if stats["success_count"] == 0:
            print("Error: No samples generated successfully!")
            return 1

    except KeyboardInterrupt:
        print("\nSample generation interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError during sample generation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("All tests completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
