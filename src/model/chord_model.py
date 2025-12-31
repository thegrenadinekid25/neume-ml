"""Main chord recognition model combining all components.

This module defines the ChordRecognitionModel which integrates audio processing,
feature extraction, conformer blocks, and chord classification heads into a unified
end-to-end model for chord recognition tasks.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .preprocessing import AudioToFeatures
from .subsampling import ConvSubsampling
from .conformer import ConformerBlock, PositionalEncoding
from .heads import ChordRecognitionHeads


@dataclass
class ChordModelConfig:
    """Configuration for the chord recognition model.

    Attributes:
        sample_rate: Audio sample rate in Hz. Default: 44100.
        hop_length: Number of samples between successive frames. Default: 512.
        n_bins: Number of frequency bins in CQT representation. Default: 72.
        bins_per_octave: Number of bins per octave in CQT. Default: 12.
        fmin: Minimum frequency in Hz. Default: 65.41 (C2).
        d_model: Dimensionality of model embeddings. Default: 256.
        num_conformer_blocks: Number of Conformer blocks to stack. Default: 6.
        num_attention_heads: Number of attention heads in self-attention. Default: 8.
        conv_kernel_size: Kernel size for convolutional layers. Default: 31.
        ff_expansion_factor: Expansion factor for feed-forward layers. Default: 4.
        dropout: Dropout probability. Default: 0.1.
        num_chord_types: Number of chord type classes. Default: 32.
        num_pitch_classes: Number of pitch classes (typically 12). Default: 12.
    """

    # Audio processing configuration
    sample_rate: int = 44100
    hop_length: int = 512
    n_bins: int = 72
    bins_per_octave: int = 12
    fmin: float = 65.41

    # Model architecture configuration
    d_model: int = 256
    num_conformer_blocks: int = 6
    num_attention_heads: int = 8
    conv_kernel_size: int = 31
    ff_expansion_factor: int = 4
    dropout: float = 0.1

    # Classification configuration
    num_chord_types: int = 32
    num_pitch_classes: int = 12

    @property
    def subsampling_factor(self) -> int:
        """Factor by which ConvSubsampling reduces sequence length.

        Returns:
            int: Subsampling factor (default 4).
        """
        return 4


class ChordRecognitionModel(nn.Module):
    """End-to-end chord recognition model.

    Combines audio feature extraction, conformer blocks, and chord classification
    heads to predict chord types, root notes, and bass notes from raw audio.

    Attributes:
        config: Model configuration.
        audio_to_features: Audio preprocessing and feature extraction.
        subsampling: Convolutional subsampling layer.
        pos_encoding: Positional encoding module.
        conformer_blocks: Stack of Conformer blocks.
        heads: Chord classification heads.
    """

    def __init__(self, config: Optional[ChordModelConfig] = None) -> None:
        """Initialize the chord recognition model.

        Args:
            config: ChordModelConfig instance. If None, uses default configuration.
        """
        super().__init__()

        self.config = config if config is not None else ChordModelConfig()

        # Audio feature extraction
        self.audio_to_features = AudioToFeatures(
            sample_rate=self.config.sample_rate,
            hop_length=self.config.hop_length,
            n_bins=self.config.n_bins,
            bins_per_octave=self.config.bins_per_octave,
            fmin=self.config.fmin,
        )

        # Subsampling to reduce sequence length
        self.subsampling = ConvSubsampling(
            input_dim=self.config.n_bins,
            output_dim=self.config.d_model,
            dropout=self.config.dropout,
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=self.config.d_model,
            dropout=self.config.dropout,
        )

        # Stack of Conformer blocks
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=self.config.d_model,
                    num_heads=self.config.num_attention_heads,
                    conv_kernel_size=self.config.conv_kernel_size,
                    ff_expansion_factor=self.config.ff_expansion_factor,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.num_conformer_blocks)
            ]
        )

        # Classification heads
        self.heads = ChordRecognitionHeads(
            d_model=self.config.d_model,
            num_chord_types=self.config.num_chord_types,
            num_pitch_classes=self.config.num_pitch_classes,
        )

    def forward(
        self,
        audio: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the chord recognition model.

        Processing pipeline:
        1. Audio to CQT features
        2. Convolutional subsampling
        3. Positional encoding
        4. Conformer blocks
        5. Mean pooling over sequence
        6. Classification heads

        Args:
            audio: Raw audio input tensor of shape (batch_size, num_samples).
            return_features: If True, also returns intermediate features and CQT.
                Default: False.

        Returns:
            Dictionary containing:
                - 'chord_type': Logits for chord type classification,
                    shape (batch_size, num_chord_types).
                - 'root': Logits for root note classification,
                    shape (batch_size, num_pitch_classes).
                - 'bass_note': Logits for bass note classification,
                    shape (batch_size, num_pitch_classes).
                - 'features': Conformer output features if return_features=True,
                    shape (batch_size, d_model).
                - 'cqt': CQT representation if return_features=True,
                    shape (batch_size, n_bins, time_steps).

        Raises:
            ValueError: If audio tensor has invalid shape.
        """
        if audio.dim() != 2:
            raise ValueError(
                f"Expected audio tensor of shape (batch_size, num_samples), "
                f"got shape {audio.shape}"
            )

        # Extract CQT features
        cqt = self.audio_to_features(audio)  # (batch_size, n_bins, time_steps)

        # Subsampling
        x = self.subsampling(cqt)  # (batch_size, time_steps_sub, d_model)

        # Positional encoding
        x = self.pos_encoding(x)  # (batch_size, time_steps_sub, d_model)

        # Conformer blocks
        for conformer_block in self.conformer_blocks:
            x = conformer_block(x)  # (batch_size, time_steps_sub, d_model)

        # Mean pooling over sequence dimension
        features = x.mean(dim=1)  # (batch_size, d_model)

        # Classification heads
        output = self.heads(features)

        if return_features:
            output['features'] = features
            output['cqt'] = cqt

        return output

    def predict(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get argmax predictions from the model.

        Performs forward pass and returns predicted class indices for each
        classification head.

        Args:
            audio: Raw audio input tensor of shape (batch_size, num_samples).

        Returns:
            Dictionary containing predicted class indices:
                - 'chord_type': Predicted chord type indices,
                    shape (batch_size,).
                - 'root': Predicted root note indices,
                    shape (batch_size,).
                - 'bass_note': Predicted bass note indices,
                    shape (batch_size,).

        Raises:
            ValueError: If audio tensor has invalid shape.
        """
        with torch.no_grad():
            logits = self.forward(audio, return_features=False)

        predictions = {
            'chord_type': torch.argmax(logits['chord_type'], dim=-1),
            'root': torch.argmax(logits['root'], dim=-1),
            'bass_note': torch.argmax(logits['bass_note'], dim=-1),
        }

        return predictions

    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters.

        Returns:
            int: Total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    d_model: int = 256,
    num_blocks: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    **kwargs,
) -> ChordRecognitionModel:
    """Factory function to create a chord recognition model.

    Creates a ChordRecognitionModel with specified configuration parameters.
    Additional keyword arguments are passed to ChordModelConfig.

    Args:
        d_model: Model embedding dimensionality. Default: 256.
        num_blocks: Number of Conformer blocks. Default: 6.
        num_heads: Number of attention heads. Default: 8.
        dropout: Dropout probability. Default: 0.1.
        **kwargs: Additional arguments passed to ChordModelConfig
            (e.g., sample_rate, hop_length, num_chord_types, etc.).

    Returns:
        ChordRecognitionModel: Initialized chord recognition model.

    Example:
        >>> model = create_model(d_model=512, num_blocks=12, num_heads=16)
        >>> audio = torch.randn(2, 44100 * 10)  # 10 seconds at 44.1kHz
        >>> output = model(audio)
    """
    config = ChordModelConfig(
        d_model=d_model,
        num_conformer_blocks=num_blocks,
        num_attention_heads=num_heads,
        dropout=dropout,
        **kwargs,
    )

    return ChordRecognitionModel(config)
