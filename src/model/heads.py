"""Classification heads for the chord recognition model.

This module provides various classification heads used in the chord recognition
pipeline, including a general-purpose ClassificationHead and a specialized
ChordRecognitionHeads module that handles chord type, root note, and bass note
prediction.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """A standard classification head with layer normalization and dropout.

    This head applies layer normalization, dropout, and two linear layers with
    GELU activation to process pooled features and produce class logits.

    Attributes:
        d_model: Dimension of input features.
        num_classes: Number of output classes.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1) -> None:
        """Initialize the classification head.

        Args:
            d_model: Dimension of input features.
            num_classes: Number of output classes for classification.
            dropout: Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classification head.

        Args:
            x: Input tensor of shape [batch_size, d_model].

        Returns:
            Output logits tensor of shape [batch_size, num_classes].
        """
        x = self.norm(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class ChordRecognitionHeads(nn.Module):
    """Multi-task classification heads for chord recognition.

    This module contains three specialized classification heads for predicting:
    - Chord type (major, minor, dominant, etc.)
    - Root note (pitch class 0-11)
    - Bass note (pitch class 0-11)

    Attributes:
        d_model: Dimension of input features.
        num_chord_types: Number of chord type classes.
        num_pitch_classes: Number of pitch classes (12 for Western music).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_chord_types: int = 32,
        num_pitch_classes: int = 12,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the chord recognition heads.

        Args:
            d_model: Dimension of input features (e.g., encoder output dimension).
            num_chord_types: Number of chord type classes. Defaults to 32.
            num_pitch_classes: Number of pitch classes. Defaults to 12.
            dropout: Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.chord_type_head = ClassificationHead(
            d_model, num_chord_types, dropout
        )
        self.root_head = ClassificationHead(d_model, num_pitch_classes, dropout)
        self.bass_head = ClassificationHead(d_model, num_pitch_classes, dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through all classification heads.

        Args:
            x: Pooled feature tensor of shape [batch_size, d_model].

        Returns:
            Dictionary containing:
                - 'chord_type': Chord type logits [batch_size, num_chord_types]
                - 'root': Root note logits [batch_size, num_pitch_classes]
                - 'bass_note': Bass note logits [batch_size, num_pitch_classes]
        """
        return {
            "chord_type": self.chord_type_head(x),
            "root": self.root_head(x),
            "bass_note": self.bass_head(x),
        }
