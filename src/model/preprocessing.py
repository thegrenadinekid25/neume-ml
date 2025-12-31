"""Audio preprocessing pipeline for chord recognition model.

This module provides CQT spectrogram computation and normalization for audio
preprocessing in the chord recognition neural network.
"""

import numpy as np
import torch
import torch.nn as nn
import librosa


class CQTProcessor(nn.Module):
    """Computes Constant-Q Transform (CQT) spectrograms from audio waveforms.

    The CQT is a logarithmically-spaced time-frequency representation that is
    particularly well-suited for music analysis, as it aligns with musical pitch
    perception and the musical scale.

    Attributes:
        sample_rate: Sample rate of the input audio in Hz.
        hop_length: Number of samples between successive frames.
        n_bins: Total number of frequency bins in the CQT.
        bins_per_octave: Number of bins per musical octave.
        fmin: Minimum frequency in Hz (C2 = 65.41 Hz).
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        n_bins: int = 72,
        bins_per_octave: int = 12,
        fmin: float = 65.41,
    ) -> None:
        """Initializes the CQT processor.

        Args:
            sample_rate: Sample rate of the input audio in Hz. Default: 44100.
            hop_length: Number of samples between successive frames. Default: 512.
            n_bins: Total number of frequency bins in the CQT. Default: 72 (6 octaves).
            bins_per_octave: Number of bins per musical octave. Default: 12 (semitones).
            fmin: Minimum frequency in Hz. Default: 65.41 (C2).
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.fmin = fmin

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Computes CQT spectrograms for a batch of audio waveforms.

        Applies Constant-Q Transform to the input waveforms, converts to dB scale,
        and normalizes to the range [0, 1].

        Args:
            waveform: Input audio waveform of shape [batch, samples].

        Returns:
            CQT spectrogram of shape [batch, n_bins, time], normalized to [0, 1].

        Raises:
            ValueError: If input waveform is not 2D or has incompatible shape.
        """
        if waveform.dim() != 2:
            raise ValueError(
                f"Expected 2D input tensor [batch, samples], got shape {waveform.shape}"
            )

        batch_size = waveform.shape[0]
        device = waveform.device
        dtype = waveform.dtype

        # Process each sample in the batch
        cqt_specs = []

        for i in range(batch_size):
            # Convert to numpy and move to CPU for librosa processing
            audio_np = waveform[i].cpu().numpy()

            # Compute CQT
            cqt = librosa.cqt(
                y=audio_np,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                fmin=self.fmin,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
            )

            # Convert to magnitude
            cqt_mag = np.abs(cqt)

            # Convert to dB scale
            cqt_db = librosa.power_to_db(cqt_mag ** 2, ref=np.max)

            # Normalize to [0, 1]
            # Handle edge case where all values are the same
            min_val = cqt_db.min()
            max_val = cqt_db.max()

            if max_val - min_val > 1e-6:
                cqt_normalized = (cqt_db - min_val) / (max_val - min_val)
            else:
                cqt_normalized = np.zeros_like(cqt_db)

            cqt_specs.append(torch.from_numpy(cqt_normalized))

        # Stack into batch and move to original device/dtype
        cqt_batch = torch.stack(cqt_specs).to(device=device, dtype=dtype)

        return cqt_batch


class CQTNormalization(nn.Module):
    """Instance normalization for CQT spectrograms.

    Applies instance normalization independently to each sample in the batch,
    which helps normalize the dynamic range of the spectrograms while preserving
    relative frequency relationships.

    Attributes:
        n_bins: Number of frequency bins (channels for normalization).
    """

    def __init__(self, n_bins: int = 72) -> None:
        """Initializes the CQT normalization layer.

        Args:
            n_bins: Number of frequency bins (treated as channels). Default: 72.
        """
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features=n_bins, affine=True)

    def forward(self, cqt_spec: torch.Tensor) -> torch.Tensor:
        """Applies instance normalization to CQT spectrograms.

        Args:
            cqt_spec: CQT spectrogram of shape [batch, n_bins, time].

        Returns:
            Normalized CQT spectrogram of shape [batch, n_bins, time].

        Raises:
            ValueError: If input is not 3D or has incompatible shape.
        """
        if cqt_spec.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor [batch, n_bins, time], got shape {cqt_spec.shape}"
            )

        return self.norm(cqt_spec)


class AudioToFeatures(nn.Module):
    """Complete audio-to-CQT-features pipeline.

    Combines CQT computation and normalization into a single end-to-end module
    for converting raw audio waveforms to normalized CQT spectrograms.

    Attributes:
        cqt_processor: CQT computation module.
        cqt_normalization: Instance normalization module.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        n_bins: int = 72,
        bins_per_octave: int = 12,
        fmin: float = 65.41,
    ) -> None:
        """Initializes the audio-to-features pipeline.

        Args:
            sample_rate: Sample rate of the input audio in Hz. Default: 44100.
            hop_length: Number of samples between successive frames. Default: 512.
            n_bins: Total number of frequency bins in the CQT. Default: 72.
            bins_per_octave: Number of bins per musical octave. Default: 12.
            fmin: Minimum frequency in Hz. Default: 65.41 (C2).
        """
        super().__init__()
        self.cqt_processor = CQTProcessor(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            fmin=fmin,
        )
        self.cqt_normalization = CQTNormalization(n_bins=n_bins)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Converts audio waveforms to normalized CQT spectrograms.

        Applies the complete preprocessing pipeline: CQT computation followed
        by instance normalization.

        Args:
            waveform: Input audio waveform of shape [batch, samples].

        Returns:
            Normalized CQT spectrogram of shape [batch, n_bins, time].

        Raises:
            ValueError: If input waveform has incompatible shape.
        """
        # Compute CQT
        cqt_spec = self.cqt_processor(waveform)

        # Apply normalization
        cqt_normalized = self.cqt_normalization(cqt_spec)

        return cqt_normalized
