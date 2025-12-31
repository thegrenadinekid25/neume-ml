"""Metrics tracking and computation for training.

This module provides utilities for tracking and computing metrics during training
and validation, including accuracy calculations and metric aggregation.
"""

from typing import Dict, Optional
from collections import defaultdict

import torch
import torch.nn as nn


class MetricsTracker:
    """Tracks and aggregates metrics during training and validation.

    Maintains running statistics of loss and accuracy metrics, with support for
    per-batch updates and epoch-level aggregation.

    Attributes:
        _metrics: Dictionary of metric values for current epoch.
        _counts: Dictionary of sample counts for averaging.
    """

    def __init__(self) -> None:
        """Initialize the metrics tracker."""
        self.reset()

    def reset(self) -> None:
        """Reset all tracked metrics for a new epoch."""
        self._metrics: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)

    def update(
        self,
        loss_dict: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        """Update metrics with batch results.

        Args:
            loss_dict: Dictionary with loss values from criterion.
                Expected keys: 'total', 'chord_type_loss', 'root_loss', 'bass_note_loss'.
            outputs: Dictionary with model predictions (logits).
                Expected keys: 'chord_type', 'root', 'bass_note'.
            targets: Dictionary with ground truth labels.
                Expected keys: 'chord_type', 'root', 'bass_note'.
            batch_size: Number of samples in the batch.
        """
        # Update losses (weighted by batch size for correct averaging)
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                self._metrics[f'{key}'] += value.item() * batch_size
            else:
                self._metrics[f'{key}'] += value * batch_size

        # Compute and update accuracies
        for task_name in ['chord_type', 'root', 'bass_note']:
            if task_name in outputs and task_name in targets:
                preds = torch.argmax(outputs[task_name], dim=-1)
                correct = (preds == targets[task_name]).sum().item()
                self._metrics[f'{task_name}_acc'] += correct

        # Update sample count
        self._counts['samples'] += batch_size

    def get_metrics(self) -> Dict[str, float]:
        """Get averaged metrics for current epoch.

        Returns:
            Dictionary with averaged metric values. Losses are averaged by sample count,
            accuracies are converted to percentages.
        """
        if self._counts['samples'] == 0:
            return {}

        metrics = {}
        num_samples = self._counts['samples']

        # Average losses
        loss_keys = ['total', 'chord_type_loss', 'root_loss', 'bass_note_loss']
        for key in loss_keys:
            if key in self._metrics:
                metrics[key] = self._metrics[key] / num_samples

        # Convert accuracies to percentages
        acc_keys = ['chord_type_acc', 'root_acc', 'bass_note_acc']
        for key in acc_keys:
            if key in self._metrics:
                metrics[key] = (self._metrics[key] / num_samples) * 100

        return metrics

    def summary(self) -> str:
        """Get formatted summary of current metrics.

        Returns:
            String representation of metrics for logging.
        """
        metrics = self.get_metrics()
        if not metrics:
            return "No metrics recorded"

        parts = []
        if 'total' in metrics:
            parts.append(f"Loss: {metrics['total']:.4f}")
        if 'chord_type_acc' in metrics:
            parts.append(f"Chord Acc: {metrics['chord_type_acc']:.2f}%")
        if 'root_acc' in metrics:
            parts.append(f"Root Acc: {metrics['root_acc']:.2f}%")
        if 'bass_note_acc' in metrics:
            parts.append(f"Bass Acc: {metrics['bass_note_acc']:.2f}%")

        return ", ".join(parts)


def compute_topk_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    k: int = 1,
) -> float:
    """Compute top-k accuracy.

    Computes the fraction of samples where the target class is in the top-k
    predictions from the model.

    Args:
        outputs: Model output logits of shape (batch_size, num_classes).
        targets: Ground truth class indices of shape (batch_size,).
        k: Number of top predictions to consider. Default: 1 (standard accuracy).

    Returns:
        Top-k accuracy as a percentage (0-100).

    Raises:
        ValueError: If k is invalid or larger than number of classes.
    """
    if outputs.dim() != 2:
        raise ValueError(f"outputs must be 2D, got shape {outputs.shape}")
    if targets.dim() != 1:
        raise ValueError(f"targets must be 1D, got shape {targets.shape}")

    num_classes = outputs.shape[-1]
    if k > num_classes:
        raise ValueError(
            f"k ({k}) cannot be larger than number of classes ({num_classes})"
        )
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    batch_size = outputs.shape[0]

    # Get top-k predictions
    _, top_preds = torch.topk(outputs, k, dim=-1)  # (batch_size, k)

    # Check if targets are in top-k predictions
    targets_expanded = targets.unsqueeze(-1)  # (batch_size, 1)
    correct = torch.any(top_preds == targets_expanded, dim=-1)  # (batch_size,)

    accuracy = correct.sum().item() / batch_size * 100

    return accuracy


def compute_confusion_matrix(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """Compute confusion matrix.

    Computes the confusion matrix for multi-class classification.

    Args:
        outputs: Model output logits of shape (batch_size, num_classes).
        targets: Ground truth class indices of shape (batch_size,).
        num_classes: Number of classes. If None, inferred from outputs.

    Returns:
        Confusion matrix of shape (num_classes, num_classes) where element [i, j]
        represents the number of samples with true class i predicted as class j.
    """
    if outputs.dim() != 2:
        raise ValueError(f"outputs must be 2D, got shape {outputs.shape}")
    if targets.dim() != 1:
        raise ValueError(f"targets must be 1D, got shape {targets.shape}")

    if num_classes is None:
        num_classes = outputs.shape[-1]

    preds = torch.argmax(outputs, dim=-1)

    # Create confusion matrix
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets, preds):
        confusion[t, p] += 1

    return confusion


def compute_per_class_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Compute per-class precision, recall, and F1-score.

    Args:
        outputs: Model output logits of shape (batch_size, num_classes).
        targets: Ground truth class indices of shape (batch_size,).
        num_classes: Number of classes. If None, inferred from outputs.

    Returns:
        Dictionary with per-class metrics:
            - 'precision': Tensor of shape (num_classes,).
            - 'recall': Tensor of shape (num_classes,).
            - 'f1': Tensor of shape (num_classes,).
            - 'support': Tensor of shape (num_classes,) with sample counts per class.
    """
    if num_classes is None:
        num_classes = outputs.shape[-1]

    confusion = compute_confusion_matrix(outputs, targets, num_classes)

    # True positives, false positives, false negatives
    tp = torch.diag(confusion).float()
    fp = confusion.sum(dim=0).float() - tp
    fn = confusion.sum(dim=1).float() - tp

    # Precision and recall
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)

    # F1-score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Support (number of samples per class)
    support = confusion.sum(dim=1).float()

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
    }
