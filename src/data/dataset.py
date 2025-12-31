"""
PyTorch Dataset for chord recognition from audio files.

Loads individual JSON metadata files paired with WAV audio files.
"""

import json
import glob
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

from src.model.labels import CHORD_TYPE_TO_IDX, get_num_chord_types, get_num_pitch_classes


class ChordDataset(Dataset):
    """
    PyTorch Dataset for chord recognition.

    Loads audio-metadata pairs where each sample consists of:
    - sample_NNNNNN.json: metadata with chord_type, root, bass_note, filename
    - sample_NNNNNN.wav: audio file (44.1kHz mono)
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 44100,
        max_duration: Optional[float] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to directory containing JSON and WAV files
            sample_rate: Target sample rate (default 44100)
            max_duration: Maximum duration in seconds (None for no limit)
            transform: Optional audio transformation function
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.transform = transform

        # Calculate max samples if max_duration is specified
        self.max_samples = (
            int(max_duration * sample_rate) if max_duration else None
        )

        # Load metadata from JSON files
        self.samples = self._load_metadata()

    def _load_metadata(self) -> List[Dict]:
        """
        Load metadata from all JSON files in the data directory.

        Returns:
            List of metadata dictionaries with added chord_type_idx
        """
        samples = []

        # Find all JSON files
        json_files = sorted(glob.glob(str(self.data_dir / "*.json")))

        for json_path in json_files:
            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)

                # Convert chord_type string to index
                chord_type = metadata.get('chord_type')
                if chord_type not in CHORD_TYPE_TO_IDX:
                    print(f"Warning: Unknown chord_type '{chord_type}' in {json_path}")
                    continue

                metadata['chord_type_idx'] = CHORD_TYPE_TO_IDX[chord_type]

                # Validate required fields
                required_fields = ['chord_type', 'root', 'bass_note', 'filename']
                if not all(field in metadata for field in required_fields):
                    print(f"Warning: Missing required fields in {json_path}")
                    continue

                # Verify audio file exists
                audio_path = self.data_dir / metadata['filename']
                if not audio_path.exists():
                    print(f"Warning: Audio file not found: {audio_path}")
                    continue

                samples.append(metadata)

            except Exception as e:
                print(f"Error loading {json_path}: {e}")
                continue

        if not samples:
            raise ValueError(f"No valid samples found in {self.data_dir}")

        return samples

    def _load_audio(self, path: Path) -> np.ndarray:
        """
        Load audio from file using soundfile.

        Args:
            path: Path to audio file

        Returns:
            Audio as numpy array (mono)
        """
        audio, sr = sf.read(path, dtype='float32')

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if necessary
        if sr != self.sample_rate:
            # Simple linear interpolation for resampling
            num_samples = int(len(audio) * self.sample_rate / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, num_samples),
                np.arange(len(audio)),
                audio
            )

        return audio

    def _pad_or_truncate(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad or randomly truncate audio to target length.

        Args:
            audio: Input audio array
            target_length: Target number of samples

        Returns:
            Processed audio of exact target_length
        """
        if len(audio) >= target_length:
            # Randomly crop to target length
            start = np.random.randint(0, len(audio) - target_length + 1)
            return audio[start : start + target_length]
        else:
            # Pad with zeros
            pad_amount = target_length - len(audio)
            return np.pad(audio, (0, pad_amount), mode='constant', constant_values=0)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with audio and chord labels
        """
        metadata = self.samples[idx]

        # Load audio
        audio_path = self.data_dir / metadata['filename']
        audio = self._load_audio(audio_path)

        # Pad or truncate to max_samples if specified
        if self.max_samples is not None:
            audio = self._pad_or_truncate(audio, self.max_samples)

        # Apply optional transform
        if self.transform is not None:
            audio = self.transform(audio)

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()

        return {
            'audio': audio_tensor,
            'chord_type': torch.tensor(metadata['chord_type_idx'], dtype=torch.long),
            'root': torch.tensor(metadata['root'], dtype=torch.long),
            'bass_note': torch.tensor(metadata['bass_note'], dtype=torch.long),
            'metadata': metadata,
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse frequency weights for chord types.

        Useful for imbalanced datasets with weighted loss functions.

        Returns:
            Tensor of shape [num_chord_types] with inverse frequency weights
        """
        # Count occurrences of each chord type
        num_classes = get_num_chord_types()
        counts = torch.zeros(num_classes)

        for sample in self.samples:
            chord_idx = sample['chord_type_idx']
            counts[chord_idx] += 1

        # Compute inverse frequencies
        total = counts.sum()
        weights = total / (counts + 1e-8)  # Add epsilon to avoid division by zero

        # Normalize so mean weight is 1
        weights = weights / weights.mean()

        return weights


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for variable-length audio samples.

    Pads audio tensors to the maximum length in the batch.

    Args:
        batch: List of samples from ChordDataset

    Returns:
        Dictionary with batched tensors
    """
    # Find max audio length in batch
    max_length = max(sample['audio'].shape[0] for sample in batch)

    # Pad all audio to max length
    batch_audio = []
    batch_chord_type = []
    batch_root = []
    batch_bass_note = []
    batch_metadata = []

    for sample in batch:
        audio = sample['audio']

        # Pad audio to max_length
        if audio.shape[0] < max_length:
            pad_amount = max_length - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, pad_amount), mode='constant', value=0)

        batch_audio.append(audio)
        batch_chord_type.append(sample['chord_type'])
        batch_root.append(sample['root'])
        batch_bass_note.append(sample['bass_note'])
        batch_metadata.append(sample['metadata'])

    return {
        'audio': torch.stack(batch_audio),  # [batch_size, max_length]
        'chord_type': torch.stack(batch_chord_type),  # [batch_size]
        'root': torch.stack(batch_root),  # [batch_size]
        'bass_note': torch.stack(batch_bass_note),  # [batch_size]
        'metadata': batch_metadata,  # list of dicts
    }


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_duration: Optional[float] = None,
    train_transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_duration: Maximum audio duration in seconds
        train_transform: Optional augmentation transform for training data

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = ChordDataset(
        train_dir,
        max_duration=max_duration,
        transform=train_transform,
    )

    val_dataset = ChordDataset(
        val_dir,
        max_duration=max_duration,
        transform=None,  # No augmentation for validation
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
