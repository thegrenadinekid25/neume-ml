# Training Guide - Chord Recognition Model

This guide covers using the `Trainer` class to train the `ChordRecognitionModel` for chord recognition tasks.

## Overview

The training module provides:

- **TrainingConfig**: Dataclass for centralized training hyperparameter management
- **Trainer**: Full training engine with validation, checkpointing, and early stopping
- **MetricsTracker**: Metric aggregation for losses and accuracies
- **Utility functions**: Top-k accuracy, confusion matrices, per-class metrics

## Quick Start

```python
from src.model.chord_model import ChordRecognitionModel
from src.model.loss import ChordRecognitionLoss
from src.training import Trainer, TrainingConfig
from torch.utils.data import DataLoader

# Initialize model
model = ChordRecognitionModel()

# Create loss function
criterion = ChordRecognitionLoss(
    chord_type_weight=1.0,
    root_weight=1.0,
    bass_weight=0.5,
    label_smoothing=0.1,
)

# Prepare data loaders
train_loader = DataLoader(train_dataset, batch_size=16)
val_loader = DataLoader(val_dataset, batch_size=16)

# Configure training
config = TrainingConfig(
    learning_rate=1e-4,
    num_epochs=100,
    warmup_epochs=5,
    scheduler='cosine',
)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    criterion=criterion,
)

# Train
history = trainer.train()

# Save history
trainer.export_history('history.json')
```

## TrainingConfig

Configuration dataclass for all training parameters:

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | Initial learning rate for AdamW optimizer |
| `weight_decay` | 0.01 | L2 regularization coefficient |
| `batch_size` | 16 | Training batch size |
| `num_epochs` | 100 | Total training epochs |
| `warmup_epochs` | 5 | Epochs for LR warmup (used with schedulers) |

### Scheduler Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scheduler` | 'cosine' | LR scheduler type: 'cosine' or 'onecycle' |
| `gradient_clip` | 1.0 | Max gradient norm for clipping |
| `label_smoothing` | 0.1 | Label smoothing in loss function |

### Checkpoint & Stopping

| Parameter | Default | Description |
|-----------|---------|-------------|
| `checkpoint_dir` | 'checkpoints' | Directory for saving checkpoints |
| `save_every` | 5 | Save checkpoint every N epochs |
| `patience` | 15 | Early stopping patience (epochs) |
| `min_delta` | 0.001 | Minimum validation loss improvement |

### Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `log_every` | 50 | Log metrics every N batches |
| `device` | auto | 'cuda' if available, else 'cpu' |

## Trainer Class

Main training engine with the following interface:

### Initialization

```python
trainer = Trainer(
    model=model,                    # nn.Module to train
    train_loader=train_loader,      # Training DataLoader
    val_loader=val_loader,          # Validation DataLoader
    config=config,                  # TrainingConfig (optional)
    criterion=criterion,            # Loss function (optional)
)
```

### Training Methods

#### `train() -> Dict[str, List[float]]`

Runs the complete training loop with:
- Per-epoch training
- Validation after each epoch
- Learning rate scheduling
- Checkpoint saving
- Early stopping
- Training history tracking

Returns training history with keys:
- `train_loss`: List of training losses
- `val_loss`: List of validation losses
- `val_chord_type_acc`: Validation chord type accuracies
- `val_root_acc`: Validation root note accuracies
- `val_bass_note_acc`: Validation bass note accuracies
- `learning_rate`: List of learning rates per epoch

#### `train_epoch() -> Dict[str, float]`

Runs a single training epoch. Returns metrics dict with:
- `total_loss`: Average total loss
- `chord_type_loss`: Chord type component
- `root_loss`: Root note component
- `bass_note_loss`: Bass note component
- `learning_rate`: Current learning rate

#### `validate() -> Dict[str, float]`

Runs validation on full validation set. Returns:
- `total_loss`: Average validation loss
- `chord_type_loss`, `root_loss`, `bass_note_loss`: Component losses
- `chord_type_acc`: Chord type accuracy (%)
- `root_acc`: Root note accuracy (%)
- `bass_note_acc`: Bass note accuracy (%)

### Checkpoint Management

#### `save_checkpoint(path: str)`

Saves:
- Model state dict
- Optimizer state
- Scheduler state
- Training history
- Best validation loss
- Configuration

```python
trainer.save_checkpoint('checkpoints/my_checkpoint.pt')
```

#### `load_checkpoint(path: str)`

Restores from checkpoint to resume training:

```python
trainer.load_checkpoint('checkpoints/best_model.pt')
history = trainer.train()  # Resume from saved state
```

### Utilities

#### `export_history(path: str)`

Export training history to JSON:

```python
trainer.export_history('history.json')
```

## Data Format

DataLoader batches should yield tuples of `(audio, targets)`:

```python
# audio: torch.Tensor of shape (batch_size, num_samples)
# targets: Dict with keys 'chord_type', 'root', 'bass_note'
#          Each value is a torch.Tensor of shape (batch_size,) with class indices

audio = torch.randn(16, 44100)  # 16 samples, 1 second at 44.1kHz
targets = {
    'chord_type': torch.randint(0, 32, (16,)),  # 32 chord types
    'root': torch.randint(0, 12, (16,)),        # 12 pitch classes
    'bass_note': torch.randint(0, 12, (16,)),   # 12 pitch classes
}
batch = (audio, targets)
```

## Learning Rate Schedulers

### Cosine Annealing (`scheduler='cosine'`)

Implements cosine annealing with:
- Linear warmup over `warmup_epochs`
- Cosine decay from `learning_rate` to `learning_rate * 1e-2`
- No step call needed per batch (only per epoch)

Good for:
- Stable training
- No momentum oscillations
- Predictable LR schedule

### OneCycle (`scheduler='onecycle'`)

Implements one-cycle learning rate policy with:
- Linear warmup to max LR
- Linear decay to final LR
- Optional momentum cycling
- Steps called per batch

Good for:
- Faster convergence
- Finding good learning rates
- Limited computation budgets

## Training Tips

### Hyperparameter Selection

1. **Learning Rate**: Start with `1e-4` and adjust:
   - Increase if training loss plateaus early
   - Decrease if training is unstable (NaN losses)

2. **Batch Size**: Balance memory and convergence:
   - Larger batches: more stable gradients, faster per-epoch
   - Smaller batches: better generalization, more gradient noise

3. **Warmup**: Use `5-10` epochs to stabilize initial training

4. **Weight Decay**: `0.01` usually works well for regularization

### Early Stopping

Configure via `patience` and `min_delta`:

```python
config = TrainingConfig(
    patience=15,       # Stop after 15 epochs without improvement
    min_delta=0.001,   # Require 0.001 improvement in validation loss
)
```

### Gradient Clipping

Set `gradient_clip > 0` to prevent exploding gradients:

```python
config = TrainingConfig(gradient_clip=1.0)  # Clip norms > 1.0
```

## Metrics Tracker

The `MetricsTracker` class handles per-batch metric updates:

```python
from src.training.metrics import MetricsTracker

tracker = MetricsTracker()
tracker.update(loss_dict, outputs, targets, batch_size)
metrics = tracker.get_metrics()
summary = tracker.summary()  # Formatted string for logging
```

## Utility Functions

### Top-K Accuracy

```python
from src.training.metrics import compute_topk_accuracy

# Check if correct class in top-3 predictions
top3_acc = compute_topk_accuracy(logits, targets, k=3)  # Returns percentage
```

### Confusion Matrix

```python
from src.training.metrics import compute_confusion_matrix

confusion = compute_confusion_matrix(logits, targets)  # (num_classes, num_classes)
```

### Per-Class Metrics

```python
from src.training.metrics import compute_per_class_metrics

metrics = compute_per_class_metrics(logits, targets)
# Returns: {'precision': ..., 'recall': ..., 'f1': ..., 'support': ...}
```

## Complete Example

See `src/training/example_usage.py` for a complete working example with dummy data.

Run with:

```bash
cd neume-ml
python -m src.training.example_usage
```

## Troubleshooting

### Training loss is NaN

- Reduce learning rate
- Check input data for NaN/Inf values
- Reduce batch size for more stable gradients

### Validation loss not improving

- Check data quality and labels
- Increase learning rate
- Reduce regularization (weight_decay)
- Ensure data loader batch format is correct

### Out of memory

- Reduce batch size
- Reduce model size (d_model, num_blocks)
- Reduce audio sequence length
- Use gradient accumulation

### Slow training

- Use mixed precision (torch.amp)
- Check DataLoader num_workers
- Profile with `torch.profiler`
- Use OneCycleLR scheduler

## References

- [PyTorch Learning Rate Scheduling](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [OneCycleLR Paper](https://arxiv.org/abs/1708.07747)
- [Label Smoothing](https://arxiv.org/abs/1512.00567)
