"""Training loop for chord recognition model.

This module provides a comprehensive training framework for the ChordRecognitionModel,
including configuration management, training/validation loops, checkpointing, and
early stopping mechanisms.
"""

import logging
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .metrics import MetricsTracker


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training the chord recognition model.

    Attributes:
        learning_rate: Initial learning rate for optimizer. Default: 1e-4.
        weight_decay: L2 regularization coefficient. Default: 0.01.
        batch_size: Batch size for training. Default: 16.
        num_epochs: Total number of training epochs. Default: 100.
        warmup_epochs: Number of epochs for learning rate warmup. Default: 5.
        scheduler: Learning rate scheduler type ('cosine' or 'onecycle'). Default: 'cosine'.
        gradient_clip: Maximum norm for gradient clipping. Default: 1.0.
        label_smoothing: Label smoothing parameter for loss. Default: 0.1.
        checkpoint_dir: Directory to save checkpoints. Default: 'checkpoints'.
        save_every: Save checkpoint every N epochs. Default: 5.
        patience: Early stopping patience (epochs without improvement). Default: 15.
        min_delta: Minimum improvement delta for early stopping. Default: 0.001.
        log_every: Log metrics every N batches. Default: 50.
        device: Device to train on ('cuda' or 'cpu'). Default: 'cuda' if available.
    """

    # Optimizer configuration
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Training schedule
    batch_size: int = 16
    num_epochs: int = 100
    warmup_epochs: int = 5

    # Learning rate scheduler
    scheduler: str = 'cosine'  # 'cosine' or 'onecycle'

    # Gradient configuration
    gradient_clip: float = 1.0

    # Loss configuration
    label_smoothing: float = 0.1

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every: int = 5

    # Early stopping
    patience: int = 15
    min_delta: float = 0.001

    # Logging
    log_every: int = 50

    # Device configuration
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.scheduler not in ('cosine', 'onecycle'):
            raise ValueError(f"scheduler must be 'cosine' or 'onecycle', got {self.scheduler}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be non-negative, got {self.warmup_epochs}")
        if self.num_epochs <= self.warmup_epochs:
            raise ValueError(
                f"num_epochs ({self.num_epochs}) must be greater than "
                f"warmup_epochs ({self.warmup_epochs})"
            )
        if self.patience < 0:
            raise ValueError(f"patience must be non-negative, got {self.patience}")
        if self.min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {self.min_delta}")
        if self.gradient_clip < 0:
            raise ValueError(f"gradient_clip must be non-negative, got {self.gradient_clip}")


class Trainer:
    """Training engine for chord recognition model.

    Manages the training loop, validation, checkpointing, and early stopping for
    the ChordRecognitionModel. Handles learning rate scheduling, gradient clipping,
    and metrics tracking.

    Attributes:
        model: The chord recognition model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Training configuration.
        optimizer: AdamW optimizer instance.
        scheduler: Learning rate scheduler instance.
        criterion: Loss function.
        device: PyTorch device (cuda/cpu).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
        criterion: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Chord recognition model to train.
            train_loader: PyTorch DataLoader for training data.
            val_loader: PyTorch DataLoader for validation data.
            config: TrainingConfig instance. If None, uses defaults.
            criterion: Loss function (ChordRecognitionLoss). If None, creates default.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config if config is not None else TrainingConfig()

        # Move model to device
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Set criterion (will be overridden if provided, but kept as reference)
        self.criterion = criterion

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()

        # Training history
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0

        # Setup checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Training config: {asdict(self.config)}")

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler based on config.

        Returns:
            LRScheduler: PyTorch learning rate scheduler instance.

        Raises:
            ValueError: If scheduler type is invalid.
        """
        if self.config.scheduler == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs - self.config.warmup_epochs,
                eta_min=self.config.learning_rate * 1e-2,
            )
        elif self.config.scheduler == 'onecycle':
            total_steps = len(self.train_loader) * self.config.num_epochs
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=self.config.warmup_epochs / self.config.num_epochs,
                anneal_strategy='cos',
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=25.0,
                final_div_factor=10000.0,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler}")

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch.

        Performs forward pass, loss computation, backward pass, and optimization step
        for all batches in the training loader. Applies gradient clipping and learning
        rate scheduling.

        Returns:
            Dict with metrics:
                - 'total_loss': Mean total loss over epoch.
                - 'chord_type_loss': Mean chord type loss.
                - 'root_loss': Mean root note loss.
                - 'bass_note_loss': Mean bass note loss.
                - 'learning_rate': Current learning rate.
        """
        self.model.train()
        self.metrics_tracker.reset()

        epoch_metrics = defaultdict(float)
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Extract audio and targets from batch
            # Support both tuple format (audio, targets_dict) and dict format from collate_fn
            if isinstance(batch, dict):
                # Dict format from ChordDataset collate_fn
                audio = batch['audio']
                targets = {
                    'chord_type': batch['chord_type'],
                    'root': batch['root'],
                    'bass_note': batch['bass_note'],
                }
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                audio, targets = batch
            else:
                raise ValueError(
                    f"Batch format not recognized. Expected dict or (audio, targets), got {type(batch)}"
                )

            # Move to device
            audio = audio.to(self.device)
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in targets.items()}

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(audio)

            # Compute loss
            if self.criterion is not None:
                loss_dict = self.criterion(outputs, targets)
            else:
                raise RuntimeError("Loss criterion not set. Provide criterion in __init__.")

            total_loss = loss_dict['total']

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

            # Optimizer step
            self.optimizer.step()

            # Update scheduler (for OneCycleLR, step is called after each batch)
            if self.config.scheduler == 'onecycle':
                self.scheduler.step()

            # Update metrics
            batch_size = audio.shape[0]
            epoch_metrics['total_loss'] += loss_dict['total'].item() * batch_size
            epoch_metrics['chord_type_loss'] += loss_dict['chord_type_loss'].item() * batch_size
            epoch_metrics['root_loss'] += loss_dict['root_loss'].item() * batch_size
            epoch_metrics['bass_note_loss'] += loss_dict['bass_note_loss'].item() * batch_size
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % self.config.log_every == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {getattr(self, 'current_epoch', '?')}, "
                    f"Batch [{batch_idx + 1}/{len(self.train_loader)}], "
                    f"Loss: {loss_dict['total'].item():.4f}, "
                    f"LR: {current_lr:.2e}"
                )

        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches * batch_size, 1)

        # Update scheduler (for CosineAnnealingLR, step is called after each epoch)
        if self.config.scheduler == 'cosine':
            self.scheduler.step()

        epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

        logger.info(
            f"Training Epoch Metrics - "
            f"Loss: {epoch_metrics['total_loss']:.4f}, "
            f"Chord: {epoch_metrics['chord_type_loss']:.4f}, "
            f"Root: {epoch_metrics['root_loss']:.4f}, "
            f"Bass: {epoch_metrics['bass_note_loss']:.4f}"
        )

        return dict(epoch_metrics)

    def validate(self) -> Dict[str, float]:
        """Run validation pass.

        Evaluates model on validation set without gradient computation.
        Computes loss and accuracy metrics.

        Returns:
            Dict with metrics:
                - 'total_loss': Mean total loss over validation set.
                - 'chord_type_loss': Mean chord type loss.
                - 'root_loss': Mean root note loss.
                - 'bass_note_loss': Mean bass note loss.
                - 'chord_type_acc': Chord type accuracy.
                - 'root_acc': Root note accuracy.
                - 'bass_note_acc': Bass note accuracy.
        """
        self.model.eval()
        self.metrics_tracker.reset()

        val_metrics = defaultdict(float)
        num_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Extract audio and targets from batch
                # Support both tuple format (audio, targets_dict) and dict format from collate_fn
                if isinstance(batch, dict):
                    # Dict format from ChordDataset collate_fn
                    audio = batch['audio']
                    targets = {
                        'chord_type': batch['chord_type'],
                        'root': batch['root'],
                        'bass_note': batch['bass_note'],
                    }
                elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                    audio, targets = batch
                else:
                    raise ValueError(
                        f"Batch format not recognized. Expected dict or (audio, targets), got {type(batch)}"
                    )

                # Move to device
                audio = audio.to(self.device)
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                          for k, v in targets.items()}

                # Forward pass
                outputs = self.model(audio)

                # Compute loss
                if self.criterion is not None:
                    loss_dict = self.criterion(outputs, targets)
                else:
                    raise RuntimeError("Loss criterion not set. Provide criterion in __init__.")

                batch_size = audio.shape[0]
                val_metrics['total_loss'] += loss_dict['total'].item() * batch_size
                val_metrics['chord_type_loss'] += loss_dict['chord_type_loss'].item() * batch_size
                val_metrics['root_loss'] += loss_dict['root_loss'].item() * batch_size
                val_metrics['bass_note_loss'] += loss_dict['bass_note_loss'].item() * batch_size

                # Compute accuracies
                chord_type_preds = torch.argmax(outputs['chord_type'], dim=-1)
                root_preds = torch.argmax(outputs['root'], dim=-1)
                bass_note_preds = torch.argmax(outputs['bass_note'], dim=-1)

                chord_type_correct = (chord_type_preds == targets['chord_type']).sum().item()
                root_correct = (root_preds == targets['root']).sum().item()
                bass_note_correct = (bass_note_preds == targets['bass_note']).sum().item()

                val_metrics['chord_type_acc'] += chord_type_correct
                val_metrics['root_acc'] += root_correct
                val_metrics['bass_note_acc'] += bass_note_correct

                num_samples += batch_size

        # Average losses
        for loss_key in ['total_loss', 'chord_type_loss', 'root_loss', 'bass_note_loss']:
            val_metrics[loss_key] /= max(num_samples, 1)

        # Convert accuracies to percentages
        for acc_key in ['chord_type_acc', 'root_acc', 'bass_note_acc']:
            val_metrics[acc_key] = (val_metrics[acc_key] / max(num_samples, 1)) * 100

        logger.info(
            f"Validation Metrics - "
            f"Loss: {val_metrics['total_loss']:.4f}, "
            f"Chord Acc: {val_metrics['chord_type_acc']:.2f}%, "
            f"Root Acc: {val_metrics['root_acc']:.2f}%, "
            f"Bass Acc: {val_metrics['bass_note_acc']:.2f}%"
        )

        return dict(val_metrics)

    def train(self) -> Dict[str, List[float]]:
        """Run full training loop with early stopping.

        Trains the model for num_epochs epochs with validation after each epoch.
        Implements early stopping based on validation loss and checkpointing of
        best model.

        Returns:
            Dict with training history:
                - 'train_loss': List of training losses per epoch.
                - 'val_loss': List of validation losses per epoch.
                - 'val_chord_type_acc': List of validation chord type accuracies.
                - 'val_root_acc': List of validation root note accuracies.
                - 'val_bass_note_acc': List of validation bass note accuracies.
                - 'learning_rate': List of learning rates per epoch.
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch + 1

            # Train epoch
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['learning_rate'].append(train_metrics['learning_rate'])

            # Validation
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['total_loss'])
            self.history['val_chord_type_acc'].append(val_metrics['chord_type_acc'])
            self.history['val_root_acc'].append(val_metrics['root_acc'])
            self.history['val_bass_note_acc'].append(val_metrics['bass_note_acc'])

            # Checkpoint saving
            if (epoch + 1) % self.config.save_every == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                self.save_checkpoint(str(checkpoint_path))
                logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Early stopping logic
            current_val_loss = val_metrics['total_loss']
            improvement = self.best_val_loss - current_val_loss

            if improvement > self.config.min_delta:
                self.best_val_loss = current_val_loss
                self.best_epoch = epoch + 1
                self.patience_counter = 0

                # Save best model
                best_path = self.checkpoint_dir / "best_model.pt"
                self.save_checkpoint(str(best_path))
                logger.info(f"New best validation loss: {current_val_loss:.4f}, saved to {best_path}")
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement for {self.patience_counter}/{self.config.patience} epochs"
                )

                if self.patience_counter >= self.config.patience:
                    logger.info(
                        f"Early stopping triggered after epoch {epoch + 1}. "
                        f"Best loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}"
                    )
                    break

        logger.info(
            f"Training completed. Best validation loss: {self.best_val_loss:.4f} "
            f"at epoch {self.best_epoch}"
        )

        return dict(self.history)

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint with optimizer and scheduler state.

        Saves the model weights, optimizer state, learning rate scheduler state,
        and training history to a file. Allows resuming training from this point.

        Args:
            path: File path where checkpoint will be saved.
        """
        checkpoint = {
            'epoch': self.current_epoch if hasattr(self, 'current_epoch') else 0,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': dict(self.history),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': asdict(self.config),
        }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint and restore training state.

        Restores model weights, optimizer state, learning rate scheduler state,
        and training history from a saved checkpoint.

        Args:
            path: File path of checkpoint to load.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
            RuntimeError: If checkpoint structure is invalid.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore model and optimizer state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore training state
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.history = defaultdict(list, checkpoint.get('history', {}))

        current_epoch = checkpoint.get('epoch', 0)
        logger.info(
            f"Checkpoint loaded from {path}. "
            f"Resume from epoch {current_epoch}, best loss: {self.best_val_loss:.4f}"
        )

    def export_history(self, path: str) -> None:
        """Export training history to JSON file.

        Saves the training history (losses, accuracies, learning rates) to a JSON
        file for later analysis and visualization.

        Args:
            path: File path where history will be saved.
        """
        history_path = Path(path)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        with open(history_path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)

        logger.info(f"Training history exported to {path}")
