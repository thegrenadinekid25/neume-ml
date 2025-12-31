"""Example usage of the Trainer for chord recognition model training.

This script demonstrates how to set up and run training with the Trainer class.
"""

from src.model.chord_model import ChordRecognitionModel, ChordModelConfig
from src.model.loss import ChordRecognitionLoss
from src.training import Trainer, TrainingConfig
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_dummy_dataloader(num_samples: int = 100, batch_size: int = 16, seq_length: int = 44100):
    """Create a dummy dataloader for demonstration.

    Args:
        num_samples: Number of samples in the dataset.
        batch_size: Batch size for DataLoader.
        seq_length: Length of audio samples.

    Returns:
        DataLoader with (audio, targets) tuples.
    """
    # Create dummy audio data: (num_samples, seq_length)
    audio = torch.randn(num_samples, seq_length)

    # Create dummy targets
    targets_dict = {
        'chord_type': torch.randint(0, 32, (num_samples,)),
        'root': torch.randint(0, 12, (num_samples,)),
        'bass_note': torch.randint(0, 12, (num_samples,)),
    }

    # Create a custom dataset to return (audio, targets_dict) tuples
    class ChordDataset(TensorDataset):
        def __init__(self, audio, targets):
            self.audio = audio
            self.targets_dict = targets

        def __len__(self):
            return len(self.audio)

        def __getitem__(self, idx):
            return self.audio[idx], {
                'chord_type': self.targets_dict['chord_type'][idx],
                'root': self.targets_dict['root'][idx],
                'bass_note': self.targets_dict['bass_note'][idx],
            }

    dataset = ChordDataset(audio, targets_dict)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    """Run training example."""
    # Create model
    model_config = ChordModelConfig(
        d_model=256,
        num_conformer_blocks=6,
        num_attention_heads=8,
        num_chord_types=32,
        num_pitch_classes=12,
    )
    model = ChordRecognitionModel(config=model_config)

    # Create loss function
    criterion = ChordRecognitionLoss(
        chord_type_weight=1.0,
        root_weight=1.0,
        bass_weight=0.5,
        label_smoothing=0.1,
    )

    # Create dummy data loaders
    train_loader = create_dummy_dataloader(num_samples=200, batch_size=16)
    val_loader = create_dummy_dataloader(num_samples=50, batch_size=16)

    # Create training config
    config = TrainingConfig(
        learning_rate=1e-4,
        weight_decay=0.01,
        batch_size=16,
        num_epochs=10,
        warmup_epochs=2,
        scheduler='cosine',
        gradient_clip=1.0,
        label_smoothing=0.1,
        checkpoint_dir='checkpoints',
        save_every=5,
        patience=5,
        min_delta=0.001,
        log_every=10,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
    )

    # Run training
    history = trainer.train()

    # Export training history
    trainer.export_history('training_history.json')

    print("Training completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Best epoch: {trainer.best_epoch}")

    # To resume training from a checkpoint:
    # trainer.load_checkpoint('checkpoints/best_model.pt')
    # history = trainer.train()


if __name__ == '__main__':
    main()
