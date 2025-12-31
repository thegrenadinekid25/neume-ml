#!/usr/bin/env python3
"""
CLI training script for chord recognition model.
"""

import argparse
import sys
from pathlib import Path

# Setup path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data.dataset import create_dataloaders
from src.model import ChordRecognitionModel, get_config
from src.training import TrainingConfig, Trainer


def print_header():
    """Print training script header."""
    print("\n" + "=" * 60)
    print("  CHORD RECOGNITION MODEL TRAINING")
    print("=" * 60 + "\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a chord recognition model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Path to training data directory"
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="Path to validation data directory"
    )

    # Audio processing arguments
    parser.add_argument(
        "--max-duration",
        type=float,
        default=4.0,
        help="Maximum audio duration in seconds (default: 4.0)"
    )

    # Model arguments
    parser.add_argument(
        "--model-size",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="Model size (default: medium)"
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer (default: 0.01)"
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (default: 1.0)"
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    # Data loading arguments
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading (default: 4)"
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (default: cuda)"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    # Print header
    print_header()

    # Parse arguments
    args = parse_arguments()

    # Validate paths
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configuration:")
    print(f"  Train dir: {train_dir}")
    print(f"  Val dir: {val_dir}")
    print(f"  Max duration: {args.max_duration}s")
    print(f"  Model size: {args.model_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Gradient clip: {args.gradient_clip}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Device: {args.device}")
    if args.resume:
        print(f"  Resuming from: {args.resume}")
    print()

    # Determine device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        device = torch.device("cpu")
        print("Using CPU\n")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        max_duration=args.max_duration,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}\n")

    # Create model
    print("Creating model...")
    model_config = get_config(args.model_size)
    model = ChordRecognitionModel(config=model_config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {args.model_size}")
    print(f"  Parameters: {num_params:,}\n")

    # Create training config
    training_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        checkpoint_dir=str(checkpoint_dir),
        device=device
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config
    )

    # Load checkpoint if resuming
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print()

    # Train
    print("Starting training...\n")
    history = trainer.train()

    # Print final results
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETED")
    print("=" * 60)
    print(f"\nFinal Results:")
    if history:
        final_epoch = len(history["train_loss"]) - 1
        print(f"  Final epoch: {final_epoch + 1}")
        print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
        if "val_loss" in history and history["val_loss"]:
            print(f"  Final val loss: {history['val_loss'][-1]:.6f}")
        if "val_accuracy" in history and history["val_accuracy"]:
            print(f"  Final val accuracy: {history['val_accuracy'][-1]:.4f}")
        if "val_precision" in history and history["val_precision"]:
            print(f"  Final val precision: {history['val_precision'][-1]:.4f}")
        if "val_recall" in history and history["val_recall"]:
            print(f"  Final val recall: {history['val_recall'][-1]:.4f}")
    print(f"\nCheckpoints saved to: {checkpoint_dir}")
    print("\n")


if __name__ == "__main__":
    main()
