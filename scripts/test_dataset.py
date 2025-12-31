#!/usr/bin/env python3
"""
Test script for dataset loading and DataLoader functionality.

Tests the ChordDataset class and collate_fn with various configurations.
Run from project root: python scripts/test_dataset.py --data-dir /path/to/data
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ChordDataset, collate_fn
from src.model.labels import format_chord_name


def test_dataset_loading(data_dir: str) -> None:
    """
    Test basic dataset loading functionality.

    Args:
        data_dir: Path to dataset directory

    Tests:
        - Create ChordDataset with max_duration=4.0
        - Verify number of samples
        - Load first sample and check shapes
        - Format chord name using format_chord_name
        - Compute and verify class weights
    """
    print("\n" + "=" * 60)
    print("Testing Dataset Loading")
    print("=" * 60)

    # Create dataset with 4 second max duration
    dataset = ChordDataset(
        data_dir=data_dir,
        sample_rate=44100,
        max_duration=4.0,
        transform=None
    )

    # Check dataset size
    num_samples = len(dataset)
    print(f"Dataset size: {num_samples} samples")
    assert num_samples > 0, "Dataset is empty!"

    # Load and inspect first sample
    sample = dataset[0]
    print(f"\nFirst sample shapes:")
    print(f"  audio shape: {sample['audio'].shape}")
    print(f"  chord_type: {sample['chord_type'].item()}")
    print(f"  root: {sample['root'].item()}")
    print(f"  bass_note: {sample['bass_note'].item()}")

    # Verify audio is correctly limited to max_duration
    max_samples = int(4.0 * 44100)
    assert sample['audio'].shape[0] == max_samples, \
        f"Audio length {sample['audio'].shape[0]} != {max_samples}"

    # Format and print chord name
    chord_type_idx = sample['chord_type'].item()
    root_idx = sample['root'].item()
    bass_idx = sample['bass_note'].item()
    chord_name = format_chord_name(chord_type_idx, root_idx, bass_idx)
    print(f"  chord name: {chord_name}")

    # Test class weights
    weights = dataset.get_class_weights()
    print(f"\nClass weights shape: {weights.shape}")
    print(f"Class weights range: [{weights.min().item():.4f}, {weights.max().item():.4f}]")
    assert weights.shape[0] > 0, "Class weights not computed"
    assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=0.01), \
        "Class weights not normalized to mean=1"

    print("\nPASS: Dataset loading test successful")


def test_dataloader(data_dir: str) -> None:
    """
    Test DataLoader with collate_fn.

    Args:
        data_dir: Path to dataset directory

    Tests:
        - Create DataLoader with batch_size=4
        - Load one batch
        - Verify batch tensor shapes and padding
    """
    print("\n" + "=" * 60)
    print("Testing DataLoader")
    print("=" * 60)

    # Create dataset and dataloader
    dataset = ChordDataset(
        data_dir=data_dir,
        sample_rate=44100,
        max_duration=4.0,
        transform=None
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Load first batch
    batch = next(iter(dataloader))

    print(f"Batch shapes:")
    print(f"  audio: {batch['audio'].shape} (batch_size, max_length)")
    print(f"  chord_type: {batch['chord_type'].shape}")
    print(f"  root: {batch['root'].shape}")
    print(f"  bass_note: {batch['bass_note'].shape}")
    print(f"  metadata: {len(batch['metadata'])} items")

    # Verify batch structure
    assert batch['audio'].shape[0] <= 4, "Batch size exceeds requested size"
    assert batch['audio'].ndim == 2, "Audio should be 2D (batch_size, length)"
    assert batch['chord_type'].shape[0] == batch['audio'].shape[0], \
        "Batch size mismatch"

    # Verify padding was applied (all audio in batch should have same length)
    assert batch['audio'].shape[1] > 0, "Audio length is zero"
    print(f"\nPASS: DataLoader test successful")


def test_variable_length(data_dir: str) -> None:
    """
    Test dataset with variable-length audio (no max_duration).

    Args:
        data_dir: Path to dataset directory

    Tests:
        - Create dataset without max_duration
        - Print sample lengths for first 10 samples
        - Verify collate_fn properly pads variable-length audio
    """
    print("\n" + "=" * 60)
    print("Testing Variable-Length Audio")
    print("=" * 60)

    # Create dataset without max_duration constraint
    dataset = ChordDataset(
        data_dir=data_dir,
        sample_rate=44100,
        max_duration=None,  # No max duration
        transform=None
    )

    # Print lengths of first 10 samples
    print(f"Sample lengths (first 10 samples):")
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        length_seconds = sample['audio'].shape[0] / 44100
        print(f"  Sample {i}: {sample['audio'].shape[0]} samples ({length_seconds:.2f}s)")

    # Create dataloader and load batch
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    batch = next(iter(dataloader))

    print(f"\nBatch with variable-length audio:")
    print(f"  All audio padded to max length in batch: {batch['audio'].shape[1]}")
    print(f"  Batch size: {batch['audio'].shape[0]}")

    # Verify all audio in batch has same shape after collation
    assert batch['audio'].ndim == 2, "Audio should be 2D after collation"
    assert batch['audio'].shape[0] > 0, "Batch is empty"
    assert batch['audio'].shape[1] > 0, "Audio length is zero"

    print(f"\nPASS: Variable-length audio test successful")


def main() -> None:
    """
    Main entry point.

    Parses command-line arguments and runs all tests.
    """
    parser = argparse.ArgumentParser(
        description="Test ChordDataset loading and DataLoader functionality"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset directory containing JSON and WAV files"
    )

    args = parser.parse_args()
    data_dir = args.data_dir

    # Verify data directory exists
    if not Path(data_dir).exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("CHORD DATASET TEST SUITE")
    print("=" * 60)
    print(f"Data directory: {data_dir}")

    # Run tests
    try:
        test_dataset_loading(data_dir)
        test_dataloader(data_dir)
        test_variable_length(data_dir)

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
