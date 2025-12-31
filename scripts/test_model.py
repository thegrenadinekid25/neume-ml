"""Test suite for chord recognition model architecture.

This script tests the ChordRecognitionModel to verify:
- Forward pass functionality
- Output shapes
- Prediction mode
- Loss computation and backpropagation
- Model configurations
- Feature extraction
- Variable length input handling
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import (
    create_model,
    ChordRecognitionLoss,
    format_chord_name,
    SMALL_CONFIG,
    MEDIUM_CONFIG,
)


def test_model_forward():
    """Test basic forward pass with correct output shapes.

    Creates a small model and verifies that forward pass produces correctly
    shaped outputs for chord type, root note, and bass note predictions.
    """
    print("Testing model forward pass...")

    # Create small model
    model = create_model(d_model=128, num_blocks=2, num_heads=4)
    model.eval()

    # Dummy audio: 2 samples, 3 seconds at 44100 Hz
    audio = torch.randn(2, 44100 * 3)

    # Forward pass
    with torch.no_grad():
        output = model(audio)

    # Verify output shapes
    assert output["chord_type"].shape == (2, 32), \
        f"Expected chord_type shape (2, 32), got {output['chord_type'].shape}"
    assert output["root"].shape == (2, 12), \
        f"Expected root shape (2, 12), got {output['root'].shape}"
    assert output["bass_note"].shape == (2, 12), \
        f"Expected bass_note shape (2, 12), got {output['bass_note'].shape}"

    print("[PASS] Output shapes correct")


def test_model_predict():
    """Test prediction mode with argmax predictions.

    Verifies that the model's predict() method returns correctly shaped
    predictions and demonstrates formatting of chord names.
    """
    print("Testing model prediction mode...")

    # Create model and set to eval mode
    model = create_model(d_model=128, num_blocks=2, num_heads=4)
    model.eval()

    # 4 samples, 2 seconds each
    audio = torch.randn(4, 44100 * 2)

    # Get predictions
    with torch.no_grad():
        predictions = model.predict(audio)

    # Verify shapes are (4,) for each output
    assert predictions["chord_type"].shape == (4,), \
        f"Expected chord_type shape (4,), got {predictions['chord_type'].shape}"
    assert predictions["root"].shape == (4,), \
        f"Expected root shape (4,), got {predictions['root'].shape}"
    assert predictions["bass_note"].shape == (4,), \
        f"Expected bass_note shape (4,), got {predictions['bass_note'].shape}"

    # Print example predictions
    print("\nExample predictions:")
    for i in range(min(2, len(predictions["chord_type"]))):
        chord_name = format_chord_name(
            predictions["chord_type"][i].item(),
            predictions["root"][i].item(),
            predictions["bass_note"][i].item(),
        )
        print(f"  Sample {i}: {chord_name}")

    print("[PASS] Predictions work")


def test_loss_function():
    """Test loss computation and backpropagation.

    Verifies that the loss function computes individual and total losses,
    and that gradients can be computed via backpropagation.
    """
    print("Testing loss function and gradients...")

    # Create model in training mode
    model = create_model(d_model=128, num_blocks=2, num_heads=4)
    model.train()

    # Create loss function
    loss_fn = ChordRecognitionLoss()

    # Random audio and targets
    audio = torch.randn(2, 44100 * 2)
    targets = {
        "chord_type": torch.randint(0, 32, (2,)),
        "root": torch.randint(0, 12, (2,)),
        "bass_note": torch.randint(0, 12, (2,)),
    }

    # Forward pass
    logits = model(audio)

    # Compute losses
    loss_dict = loss_fn(logits, targets)

    # Print individual losses
    print(f"  Chord type loss: {loss_dict['chord_type_loss'].item():.4f}")
    print(f"  Root loss: {loss_dict['root_loss'].item():.4f}")
    print(f"  Bass note loss: {loss_dict['bass_note_loss'].item():.4f}")
    print(f"  Total loss: {loss_dict['total'].item():.4f}")

    # Backward pass
    loss_dict["total"].backward()

    # Verify gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"

    print("[PASS] Loss and gradients work")


def test_model_sizes():
    """Test different model configurations.

    Creates models with SMALL and MEDIUM configs and verifies they work
    by counting parameters and doing a forward pass.
    """
    print("Testing model configurations...")

    configs = [
        ("SMALL_CONFIG", SMALL_CONFIG),
        ("MEDIUM_CONFIG", MEDIUM_CONFIG),
    ]

    for config_name, config in configs:
        model = torch.nn.Module()
        # Create model from config directly
        from src.model import ChordRecognitionModel
        model = ChordRecognitionModel(config)
        model.eval()

        # Count parameters
        num_params = model.get_num_parameters()
        print(f"  {config_name}: {num_params:,} parameters")

        # Quick forward pass
        audio = torch.randn(1, 44100 * 2)
        with torch.no_grad():
            output = model(audio)

        # Verify output exists
        assert "chord_type" in output
        assert "root" in output
        assert "bass_note" in output

    print("[PASS] All configurations work")


def test_return_features():
    """Test feature extraction mode.

    Verifies that setting return_features=True includes intermediate
    features and CQT representation in the output.
    """
    print("Testing feature extraction...")

    model = create_model(d_model=128, num_blocks=2, num_heads=4)
    model.eval()

    audio = torch.randn(2, 44100 * 2)

    # Forward pass with return_features=True
    with torch.no_grad():
        output = model.forward(audio, return_features=True)

    # Verify features are included
    assert "features" in output, "Output missing 'features'"
    assert "cqt" in output, "Output missing 'cqt'"

    # Print shapes
    print(f"  Features shape: {output['features'].shape}")
    print(f"  CQT shape: {output['cqt'].shape}")

    # Verify shapes make sense
    assert output["features"].shape == (2, 128), \
        f"Expected features shape (2, 128), got {output['features'].shape}"
    assert output["cqt"].shape[0] == 2, \
        f"Expected batch size 2 in CQT, got {output['cqt'].shape[0]}"

    print("[PASS] Feature extraction works")


def test_variable_length():
    """Test model with variable length audio inputs.

    Verifies that the model can handle audio of different lengths
    (1, 2, 3, 5 seconds).
    """
    print("Testing variable length inputs...")

    model = create_model(d_model=128, num_blocks=2, num_heads=4)
    model.eval()

    lengths = [1.0, 2.0, 3.0, 5.0]

    for length in lengths:
        num_samples = int(44100 * length)
        audio = torch.randn(1, num_samples)

        with torch.no_grad():
            output = model(audio)

        # Verify output
        assert "chord_type" in output
        assert output["chord_type"].shape == (1, 32)

        print(f"  {length:.1f}s input: OK")

    print("[PASS] Variable lengths work")


def main():
    """Run all tests and report results."""
    print("=" * 60)
    print("Neume ML - Model Architecture Tests")
    print("=" * 60)
    print()

    try:
        test_model_forward()
        print()

        test_model_predict()
        print()

        test_loss_function()
        print()

        test_model_sizes()
        print()

        test_return_features()
        print()

        test_variable_length()
        print()

        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
