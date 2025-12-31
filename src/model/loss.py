"""Loss functions for multi-task chord recognition training.

This module provides loss functions for training chord recognition models
with multiple output heads (chord type, root note, bass note).
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChordRecognitionLoss(nn.Module):
    """Multi-task loss function for chord recognition.

    Combines weighted losses from multiple chord prediction tasks:
    - Chord type classification
    - Root note classification
    - Bass note classification

    Attributes:
        chord_type_weight (float): Weight for chord type loss.
        root_weight (float): Weight for root note loss.
        bass_weight (float): Weight for bass note loss.
        label_smoothing (float): Label smoothing parameter for CrossEntropyLoss.
    """

    def __init__(
        self,
        chord_type_weight: float = 1.0,
        root_weight: float = 1.0,
        bass_weight: float = 0.5,
        label_smoothing: float = 0.1,
    ) -> None:
        """Initialize ChordRecognitionLoss.

        Args:
            chord_type_weight (float, optional): Weight for chord type loss.
                Defaults to 1.0.
            root_weight (float, optional): Weight for root note loss.
                Defaults to 1.0.
            bass_weight (float, optional): Weight for bass note loss.
                Defaults to 0.5.
            label_smoothing (float, optional): Label smoothing parameter.
                Defaults to 0.1.
        """
        super().__init__()
        self.chord_type_weight = chord_type_weight
        self.root_weight = root_weight
        self.bass_weight = bass_weight

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted multi-task loss.

        Args:
            predictions (Dict[str, torch.Tensor]): Dictionary containing logits
                for each task. Expected keys: 'chord_type', 'root', 'bass_note'.
                Each value should be a tensor of shape (batch_size, num_classes).
            targets (Dict[str, torch.Tensor]): Dictionary containing target class
                indices for each task. Expected keys: 'chord_type', 'root',
                'bass_note'. Each value should be a tensor of shape (batch_size,)
                with integer class indices.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing individual losses and
                the weighted total loss. Keys: 'chord_type_loss', 'root_loss',
                'bass_note_loss', 'total'.
        """
        # Compute individual losses
        chord_type_loss = self.criterion(
            predictions["chord_type"], targets["chord_type"]
        )
        root_loss = self.criterion(predictions["root"], targets["root"])
        bass_note_loss = self.criterion(
            predictions["bass_note"], targets["bass_note"]
        )

        # Compute weighted total loss
        total_loss = (
            self.chord_type_weight * chord_type_loss
            + self.root_weight * root_loss
            + self.bass_weight * bass_note_loss
        )

        return {
            "chord_type_loss": chord_type_loss,
            "root_loss": root_loss,
            "bass_note_loss": bass_note_loss,
            "total": total_loss,
        }


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    Focal loss applies a modulating factor to the cross entropy loss to down-weight
    easy examples and focus on hard negatives. This is particularly useful for
    training on imbalanced datasets.

    Attributes:
        gamma (float): Focusing parameter for modulating loss from hard examples.
        alpha (float): Weighting factor in [0, 1] to balance positive/negative examples.
        label_smoothing (float): Label smoothing parameter.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        label_smoothing: float = 0.1,
    ) -> None:
        """Initialize FocalLoss.

        Args:
            gamma (float, optional): Focusing parameter. Higher values focus more on
                hard negatives. Defaults to 2.0.
            alpha (float, optional): Weighting factor for balancing examples.
                Defaults to 0.25.
            label_smoothing (float, optional): Label smoothing parameter.
                Defaults to 0.1.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Implements focal loss as: alpha * (1 - pt)^gamma * CE(pt)
        where pt is the probability of the ground truth class.

        Args:
            predictions (torch.Tensor): Predicted logits of shape
                (batch_size, num_classes).
            targets (torch.Tensor): Ground truth class indices of shape
                (batch_size,).

        Returns:
            torch.Tensor: Scalar focal loss value.
        """
        # Compute cross entropy with label smoothing
        ce_loss = F.cross_entropy(
            predictions, targets, label_smoothing=self.label_smoothing, reduction="none"
        )

        # Get class probabilities
        p = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * torch.pow(1 - p, self.gamma) * ce_loss

        return focal_loss.mean()
