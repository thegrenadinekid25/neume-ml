"""Main script for generating training samples with all voicing parameters.

This module generates synthetic training data for chord recognition ML models.
It samples from the complete parameter space of chord voicing to create diverse,
realistic training examples with full metadata.

Supports optional audio augmentation to transform clean FluidSynth output into
realistic choral recordings with vibrato, pitch scatter, reverb, and dynamics.
"""

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile

from src.synthesis import FluidSynthRenderer
from src.utils.audio import load_config, resolve_soundfont_path
from src.augmentation import AugmentationPipeline, AugmentationConfig
from .voicing import (
    BASS_NOTE_TYPES,
    BASS_NOTE_WEIGHTS,
    CHORD_FREQUENCY,
    CHORD_TYPES,
    DISTRIBUTION_WEIGHTS,
    DURATION_OPTIONS,
    DURATION_WEIGHTS,
    NCT_DENSITY_WEIGHTS,
    ROOT_NAMES,
    SHELL_STRATEGIES,
    VOICING_STYLE_WEIGHTS,
    VOICING_STYLES,
    add_non_chord_tones,
    apply_voicing_style,
    distribute_voices,
    get_pitch_classes,
    select_bass_note,
    select_duration,
    select_pitches,
    validate_sample,
    weighted_choice,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SampleMetadata:
    """Metadata for a generated training sample."""

    chord_type: str
    root: int
    bass_note: int
    inversion_type: str
    shell_strategy: str
    num_parts: int
    num_distinct_pitches: int
    voicing_style: str
    duration_class: str
    duration_sec: float
    nct_density: str
    midi_notes: List[int]
    filename: str
    augmentation: Optional[Dict[str, Any]] = None  # Augmentation config if applied

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def generate_sample(seed: Optional[int] = None) -> Tuple[SampleMetadata, List[List]]:
    """
    Generate a single training sample with all voicing parameters.

    This function samples from the complete parameter space:
    - Chord type (from CHORD_FREQUENCY)
    - Root note (0-11)
    - Inversion type (from BASS_NOTE_WEIGHTS)
    - Bass note selection
    - Shell strategy
    - Number of parts (4-16)
    - Voice distribution
    - Voicing style
    - Duration
    - Non-chord tone density

    Args:
        seed: Optional random seed for reproducibility

    Returns:
        Tuple of (SampleMetadata, voice_events) where voice_events is a list
        of event lists, one per voice

    Raises:
        ValueError: If sample generation fails validation
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Sample chord type from CHORD_FREQUENCY distribution
    chord_type = weighted_choice(CHORD_FREQUENCY)

    # Sample root note uniformly (C=0 to B=11)
    root = random.randint(0, 11)

    # Sample inversion type
    inversion_type = weighted_choice(BASS_NOTE_WEIGHTS)

    # Get chord intervals and pitch classes
    intervals = CHORD_TYPES[chord_type]
    pitch_classes = get_pitch_classes(chord_type, root)

    # Select bass note based on inversion type
    bass_note = select_bass_note(chord_type, root, inversion_type, intervals)

    # Sample shell strategy
    shell_strategy = random.choice(list(SHELL_STRATEGIES.keys()))

    # Sample number of parts (4-16 voices)
    num_parts = random.randint(4, 16)

    # Determine number of distinct pitches based on DISTRIBUTION_WEIGHTS
    distribution_type = weighted_choice(DISTRIBUTION_WEIGHTS)
    num_distinct_pitches = calculate_num_distinct_pitches(
        len(pitch_classes), distribution_type
    )

    # Select pitches based on shell strategy
    selected_pitches = select_pitches(
        pitch_classes, num_distinct_pitches, shell_strategy
    )

    # Sample voicing style
    voicing_style = weighted_choice(VOICING_STYLE_WEIGHTS)

    # Distribute voices across the selected pitches
    voice_notes = distribute_voices(
        selected_pitches, bass_note, num_parts, style=voicing_style
    )

    # Sample duration
    duration_class = weighted_choice(DURATION_WEIGHTS)
    duration_sec = select_duration(duration_class)

    # Sample NCT density
    nct_density = weighted_choice(NCT_DENSITY_WEIGHTS)

    # Add non-chord tones to create voice events
    voice_events = add_non_chord_tones(voice_notes, pitch_classes, nct_density)

    # Validate sample
    metadata_dict = {
        "num_parts": num_parts,
        "expected_pitch_classes": pitch_classes,
    }
    errors = validate_sample(voice_notes, metadata_dict)
    if errors:
        error_msg = "; ".join(
            f"{e.code}: {e.message}" for e in errors
        )
        raise ValueError(f"Sample validation failed: {error_msg}")

    # Create metadata object
    metadata = SampleMetadata(
        chord_type=chord_type,
        root=root,
        bass_note=bass_note,
        inversion_type=inversion_type,
        shell_strategy=shell_strategy,
        num_parts=num_parts,
        num_distinct_pitches=num_distinct_pitches,
        voicing_style=voicing_style,
        duration_class=duration_class,
        duration_sec=duration_sec,
        nct_density=nct_density,
        midi_notes=voice_notes,
        filename="",  # Will be set when saving
    )

    return metadata, voice_events


def calculate_num_distinct_pitches(chord_size: int, distribution_type: str) -> int:
    """
    Calculate number of distinct pitches based on distribution type.

    Args:
        chord_size: Number of pitch classes in the chord
        distribution_type: One of "full", "minus_one", "minus_two", "shell"

    Returns:
        Number of distinct pitches to use
    """
    if distribution_type == "full":
        return chord_size
    elif distribution_type == "minus_one":
        return max(2, chord_size - 1)
    elif distribution_type == "minus_two":
        return max(2, chord_size - 2)
    elif distribution_type == "shell":
        return max(2, (chord_size + 1) // 2)
    else:
        return chord_size


def generate_training_set(
    num_samples: int = 100,
    output_dir: str = "data/output/training_samples",
    config_path: str = "configs/synthesis.yaml",
    seed: Optional[int] = None,
    apply_augmentation: bool = True,
    augmentation_config: Optional[Dict[str, Any]] = None,
    ir_directory: Optional[str] = None,
) -> List[SampleMetadata]:
    """
    Generate a complete training set of chord samples.

    This function:
    1. Loads the synthesis configuration
    2. Initializes the FluidSynth renderer
    3. Generates num_samples training samples
    4. Renders each sample to audio
    5. Optionally applies augmentation (reverb, vibrato, etc.)
    6. Saves audio as WAV and metadata as JSON
    7. Returns list of successfully generated samples

    Args:
        num_samples: Number of samples to generate
        output_dir: Directory to save output samples
        config_path: Path to synthesis configuration YAML
        seed: Random seed for reproducibility
        apply_augmentation: Whether to apply audio augmentation
        augmentation_config: Fixed augmentation config dict (random if None)
        ir_directory: Path to impulse response files for convolution reverb

    Returns:
        List of SampleMetadata objects for successfully generated samples

    Raises:
        FileNotFoundError: If config file not found
        RuntimeError: If soundfont cannot be loaded
    """
    # Set random seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load configuration
    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Resolve soundfont path
    soundfont_path = resolve_soundfont_path(config)
    if not Path(soundfont_path).exists():
        raise FileNotFoundError(f"Soundfont not found: {soundfont_path}")

    # Initialize FluidSynth renderer
    logger.info(f"Initializing FluidSynth renderer with {soundfont_path}")
    sample_rate = config.get("synthesis", {}).get("sample_rate", 44100)
    renderer = FluidSynthRenderer(soundfont_path, sample_rate=sample_rate)

    # Log augmentation status
    if apply_augmentation:
        logger.info("Augmentation enabled - samples will be processed with reverb, vibrato, etc.")
        if ir_directory:
            logger.info(f"Using impulse responses from: {ir_directory}")
    else:
        logger.info("Augmentation disabled - generating clean samples")

    successful_samples = []
    failed_samples = 0

    try:
        for sample_idx in range(num_samples):
            try:
                # Generate sample with unique seed per sample
                sample_seed = seed + sample_idx if seed is not None else None
                metadata, voice_events = generate_sample(seed=sample_seed)

                # Generate unique filename
                sample_id = f"{sample_idx:06d}"
                audio_filename = f"sample_{sample_id}.wav"
                json_filename = f"sample_{sample_id}.json"

                metadata.filename = audio_filename

                # Render audio with voice events
                audio = renderer.render_chord_with_events(
                    voice_events,
                    duration_sec=metadata.duration_sec,
                    velocity=80,
                    release_sec=0.5,
                )

                # Apply augmentation if enabled
                if apply_augmentation:
                    # Create augmentation config
                    if augmentation_config is not None:
                        aug_config = AugmentationConfig.from_dict(augmentation_config)
                    else:
                        # Random config for each sample
                        aug_seed = seed + sample_idx if seed is not None else None
                        aug_config = AugmentationConfig.random(seed=aug_seed)

                    # Set IR directory if provided
                    if ir_directory:
                        aug_config.ir_directory = ir_directory
                        aug_config.use_convolution_reverb = True

                    # Apply augmentation pipeline
                    pipeline = AugmentationPipeline(aug_config)
                    audio = pipeline.process_mixed(audio, sample_rate)

                    # Store augmentation config in metadata
                    metadata.augmentation = aug_config.to_dict()

                # Save audio WAV file
                audio_path = output_path / audio_filename
                soundfile.write(audio_path, audio, sample_rate)

                # Save metadata JSON file
                json_path = output_path / json_filename
                with open(json_path, "w") as f:
                    f.write(metadata.to_json())

                successful_samples.append(metadata)

                # Log progress
                if (sample_idx + 1) % 10 == 0:
                    aug_status = "augmented" if apply_augmentation else "clean"
                    logger.info(
                        f"Generated {sample_idx + 1}/{num_samples} {aug_status} samples "
                        f"({len(successful_samples)} successful, {failed_samples} failed)"
                    )

            except Exception as e:
                failed_samples += 1
                logger.warning(
                    f"Failed to generate sample {sample_idx}: {str(e)}"
                )
                continue

        # Final summary
        aug_status = "augmented" if apply_augmentation else "clean"
        logger.info(
            f"Training set generation complete: {len(successful_samples)} {aug_status} samples, "
            f"{failed_samples} failed"
        )
        logger.info(f"Output saved to {output_dir}")

    finally:
        # Clean up renderer resources
        renderer.cleanup()

    return successful_samples


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate training samples with all voicing parameters"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output/training_samples",
        help="Output directory for samples (default: data/output/training_samples)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/synthesis.yaml",
        help="Path to synthesis configuration YAML (default: configs/synthesis.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable augmentation (generate clean samples)",
    )
    parser.add_argument(
        "--augmentation-config",
        type=str,
        default=None,
        help="Path to JSON file with fixed augmentation parameters",
    )
    parser.add_argument(
        "--ir-directory",
        type=str,
        default=None,
        help="Path to directory with impulse response WAV files for convolution reverb",
    )
    parser.add_argument(
        "--augmentation-preset",
        type=str,
        choices=["minimal", "moderate", "heavy", "cathedral"],
        default=None,
        help="Use a preset augmentation configuration",
    )

    args = parser.parse_args()

    # Load augmentation config if specified
    augmentation_config = None
    if args.augmentation_config:
        with open(args.augmentation_config) as f:
            augmentation_config = json.load(f)
    elif args.augmentation_preset:
        preset_config = AugmentationConfig.from_preset(args.augmentation_preset)
        augmentation_config = preset_config.to_dict()

    try:
        generate_training_set(
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            config_path=args.config,
            seed=args.seed,
            apply_augmentation=not args.no_augmentation,
            augmentation_config=augmentation_config,
            ir_directory=args.ir_directory,
        )
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
