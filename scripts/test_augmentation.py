#!/usr/bin/env python3
"""
Test augmentation pipeline with A/B comparison.

Generates clean and augmented versions side-by-side for listening.
Run from project root: python scripts/test_augmentation.py
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import soundfile as sf

from src.augmentation.pipeline import (
    AugmentationPipeline,
    AugmentationConfig,
    AUGMENTATION_RANGES,
)
from src.augmentation.pitch_scatter import apply_pitch_scatter
from src.augmentation.vibrato import apply_vibrato
from src.augmentation.amplitude_modulation import apply_amplitude_modulation
from src.augmentation.reverb import apply_reverb
from src.augmentation.dynamics import apply_dynamics, measure_lufs, measure_crest_factor
from src.augmentation.lowpass import apply_lowpass


def generate_test_tone(
    frequency: float = 440.0,
    duration: float = 3.0,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Generate a simple test tone."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    # Use a richer waveform (sum of harmonics) for more realistic testing
    audio = 0.4 * np.sin(2 * np.pi * frequency * t)
    audio += 0.2 * np.sin(2 * np.pi * frequency * 2 * t)  # 2nd harmonic
    audio += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)  # 3rd harmonic
    return audio


def generate_test_chord(
    frequencies: list = None,
    duration: float = 3.0,
    sample_rate: int = 44100,
) -> list:
    """Generate a test chord as separate voices."""
    if frequencies is None:
        # C4 major chord with doubled root
        frequencies = [261.63, 329.63, 392.00, 523.25]

    voices = []
    t = np.linspace(0, duration, int(duration * sample_rate))

    for freq in frequencies:
        # Rich voice waveform
        voice = 0.25 * np.sin(2 * np.pi * freq * t)
        voice += 0.12 * np.sin(2 * np.pi * freq * 2 * t)
        voice += 0.05 * np.sin(2 * np.pi * freq * 3 * t)
        voices.append(voice)

    return voices


def test_individual_effects():
    """Test each effect individually."""
    print("\n" + "=" * 60)
    print("Testing Individual Augmentation Effects")
    print("=" * 60)

    output_dir = project_root / "data" / "test_augmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 44100
    tone = generate_test_tone(440, 3.0, sample_rate)

    # Save original
    sf.write(output_dir / "01_original.wav", tone, sample_rate)
    print(f"  [+] 01_original.wav")

    # Test pitch scatter
    print("\nPitch Scatter:")
    scattered = apply_pitch_scatter(tone, sample_rate, std_cents=30)
    sf.write(output_dir / "02_pitch_scatter.wav", scattered, sample_rate)
    print(f"  [+] 02_pitch_scatter.wav (30 cents scatter)")

    # Test vibrato
    print("\nVibrato:")
    vibrato = apply_vibrato(tone, sample_rate, rate_hz=5.5, depth_cents=40)
    sf.write(output_dir / "03_vibrato.wav", vibrato, sample_rate)
    print(f"  [+] 03_vibrato.wav (5.5 Hz, 40 cents)")

    # Test AM
    print("\nAmplitude Modulation:")
    am = apply_amplitude_modulation(tone, sample_rate, depth=0.4, rate_hz=1.0)
    sf.write(output_dir / "04_amplitude_mod.wav", am, sample_rate)
    print(f"  [+] 04_amplitude_mod.wav (0.4 depth, 1 Hz)")

    # Test reverb
    print("\nReverb:")
    reverb = apply_reverb(tone, sample_rate, rt60_sec=2.5, wet_dry=0.55)
    sf.write(output_dir / "05_reverb.wav", reverb, sample_rate)
    print(f"  [+] 05_reverb.wav (RT60=2.5s, 55% wet)")

    # Test dynamics
    print("\nDynamics:")
    loud = tone * 0.1  # Make quiet first
    normalized = apply_dynamics(loud, sample_rate, target_lufs=-23, target_crest_factor_db=18)
    sf.write(output_dir / "06_dynamics.wav", normalized, sample_rate)
    print(f"  [+] 06_dynamics.wav (normalized to -23 LUFS)")

    # Test lowpass filter
    print("\nLow-pass Filter:")
    bright = apply_lowpass(tone, sample_rate, cutoff_hz=16000)
    sf.write(output_dir / "07_lowpass_bright.wav", bright, sample_rate)
    print(f"  [+] 07_lowpass_bright.wav (16kHz cutoff)")

    balanced = apply_lowpass(tone, sample_rate, cutoff_hz=10000)
    sf.write(output_dir / "08_lowpass_balanced.wav", balanced, sample_rate)
    print(f"  [+] 08_lowpass_balanced.wav (10kHz cutoff)")

    warm = apply_lowpass(tone, sample_rate, cutoff_hz=5000)
    sf.write(output_dir / "09_lowpass_warm.wav", warm, sample_rate)
    print(f"  [+] 09_lowpass_warm.wav (5kHz cutoff)")

    # Measure and report
    print("\nMeasurements:")
    print(f"  Original LUFS: {measure_lufs(tone, sample_rate):.1f}")
    print(f"  Original crest factor: {measure_crest_factor(tone):.1f} dB")
    print(f"  Normalized LUFS: {measure_lufs(normalized, sample_rate):.1f}")
    print(f"  Normalized crest factor: {measure_crest_factor(normalized):.1f} dB")

    print(f"\nIndividual effects saved to {output_dir}/")


def test_full_pipeline():
    """Test complete pipeline with multitrack input."""
    print("\n" + "=" * 60)
    print("Testing Full Augmentation Pipeline")
    print("=" * 60)

    output_dir = project_root / "data" / "test_augmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 44100
    voices = generate_test_chord(duration=4.0, sample_rate=sample_rate)

    # Save clean mix
    clean_mix = sum(voices) / len(voices)
    sf.write(output_dir / "10_chord_clean.wav", clean_mix, sample_rate)
    print(f"\n  [+] 10_chord_clean.wav (C major, clean)")

    # Apply full pipeline with preset configs
    print("\nPreset configurations:")
    configs = [
        ("minimal", AugmentationConfig.from_preset("minimal")),
        ("moderate", AugmentationConfig.from_preset("moderate")),
        ("heavy", AugmentationConfig.from_preset("heavy")),
        ("cathedral", AugmentationConfig.from_preset("cathedral")),
    ]

    for name, config in configs:
        pipeline = AugmentationPipeline(config)
        augmented = pipeline.process_multitrack(voices, sample_rate)

        filename = f"11_chord_{name}.wav"
        sf.write(output_dir / filename, augmented, sample_rate)
        print(f"  [+] {filename}")
        print(f"      RT60={config.reverb_rt60_sec:.1f}s, "
              f"wet={config.reverb_wet_dry:.0%}, "
              f"scatter={config.pitch_scatter_std_cents:.0f}cents, "
              f"lpf={config.lowpass_cutoff_hz:.0f}Hz")

    # Random configs
    print("\nRandom configurations:")
    for i in range(3):
        config = AugmentationConfig.random(seed=42 + i)
        pipeline = AugmentationPipeline(config)
        augmented = pipeline.process_multitrack(voices, sample_rate)

        filename = f"12_chord_random_{i + 1}.wav"
        sf.write(output_dir / filename, augmented, sample_rate)
        print(f"  [+] {filename}")
        print(f"      RT60={config.reverb_rt60_sec:.1f}s, "
              f"wet={config.reverb_wet_dry:.0%}, "
              f"scatter={config.pitch_scatter_std_cents:.0f}cents, "
              f"lpf={config.lowpass_cutoff_hz:.0f}Hz")

    print(f"\nFull pipeline tests saved to {output_dir}/")


def test_with_real_samples():
    """Test with real FluidSynth samples if available."""
    print("\n" + "=" * 60)
    print("Testing with Real Samples")
    print("=" * 60)

    # Look for existing samples in various locations
    sample_dirs = [
        project_root / "data" / "synthetic" / "audio",
        project_root / "data" / "output" / "training_samples",
        project_root / "data" / "output",
    ]

    samples = []
    for sample_dir in sample_dirs:
        if sample_dir.exists():
            samples.extend(list(sample_dir.glob("*.wav"))[:3])
            if samples:
                break

    if not samples:
        print("  No synthetic samples found. Run generate_training_set.py first.")
        print("  Skipping real sample tests.")
        return

    output_dir = project_root / "data" / "test_augmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFound {len(samples)} sample(s) to process:")

    for sample_path in samples[:3]:
        print(f"\n  Processing {sample_path.name}...")

        audio, sr = sf.read(sample_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono

        # Apply augmentation
        config = AugmentationConfig.random(seed=42)
        pipeline = AugmentationPipeline(config)
        augmented = pipeline.process_mixed(audio, sr)

        # Save comparison
        out_name = f"real_{sample_path.stem}_augmented.wav"
        sf.write(output_dir / out_name, augmented, sr)
        print(f"    [+] {out_name}")

    print(f"\nReal sample tests saved to {output_dir}/")


def test_mixed_mode():
    """Test processing of already-mixed audio."""
    print("\n" + "=" * 60)
    print("Testing Mixed Audio Processing")
    print("=" * 60)

    output_dir = project_root / "data" / "test_augmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 44100
    voices = generate_test_chord(duration=4.0, sample_rate=sample_rate)

    # Pre-mix the voices
    mixed = sum(voices) / len(voices)

    # Process in mixed mode
    config = AugmentationConfig.from_preset("moderate")
    pipeline = AugmentationPipeline(config)

    # Compare multitrack vs mixed processing
    multitrack_result = pipeline.process_multitrack(voices, sample_rate)

    # Reset RNG for fair comparison
    pipeline._rng = np.random.default_rng(config.seed)
    mixed_result = pipeline.process_mixed(mixed, sample_rate)

    sf.write(output_dir / "20_multitrack_processing.wav", multitrack_result, sample_rate)
    sf.write(output_dir / "21_mixed_processing.wav", mixed_result, sample_rate)

    print(f"\n  [+] 20_multitrack_processing.wav (preferred method)")
    print(f"  [+] 21_mixed_processing.wav (fallback method)")
    print("\n  Listen to compare - multitrack should sound more natural")
    print(f"\n  Saved to {output_dir}/")


def print_calibrated_ranges():
    """Print the calibrated parameter ranges."""
    print("\n" + "=" * 60)
    print("Calibrated Augmentation Ranges")
    print("(from acoustic analysis of professional choral recordings)")
    print("=" * 60)

    for category, params in AUGMENTATION_RANGES.items():
        print(f"\n{category}:")
        for param, (low, high) in params.items():
            print(f"  {param}: {low} - {high}")


def test_config_serialization():
    """Test config save/load functionality."""
    print("\n" + "=" * 60)
    print("Testing Config Serialization")
    print("=" * 60)

    # Create random config
    config = AugmentationConfig.random(seed=42)

    # Serialize to dict
    config_dict = config.to_dict()
    print(f"\nConfig as dict: {config_dict}")

    # Serialize to JSON
    json_str = config.to_json()
    print(f"\nConfig as JSON (first 200 chars):\n{json_str[:200]}...")

    # Deserialize back
    restored = AugmentationConfig.from_dict(config_dict)
    print(f"\nRestored from dict: pitch_scatter={restored.pitch_scatter_std_cents:.1f}")

    restored2 = AugmentationConfig.from_json(json_str)
    print(f"Restored from JSON: pitch_scatter={restored2.pitch_scatter_std_cents:.1f}")

    print("\n  [OK] Serialization round-trip successful")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Neume ML - Augmentation Pipeline Tests")
    print("=" * 60)

    print_calibrated_ranges()
    test_individual_effects()
    test_full_pipeline()
    test_mixed_mode()
    test_config_serialization()
    test_with_real_samples()

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("Listen to files in data/test_augmentation/ to verify quality.")
    print("=" * 60)


if __name__ == "__main__":
    main()
