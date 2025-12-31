"""Convolutional subsampling module for reducing sequence length in Conformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSubsampling(nn.Module):
    """Convolutional subsampling to reduce sequence length before Conformer blocks.

    Uses two stride-2 convolutions to achieve 4x time reduction, projecting from
    input dimension (e.g., CQT bins) to model dimension.

    Args:
        input_dim: Number of input features (e.g., 72 for CQT).
        output_dim: Model dimension for projection (d_model).
        dropout: Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize ConvSubsampling module.

        Args:
            input_dim: Number of input features (n_bins).
            output_dim: Output dimension (d_model).
            dropout: Dropout probability. Defaults to 0.1.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Two stride-2 convolutions for 4x time reduction
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Calculate flattened dimension after convolutions
        # After conv1: [batch, 32, input_dim//2, time//2]
        # After conv2: [batch, 32, input_dim//4, time//4]
        conv_output_size = 32 * (input_dim // 4)

        # Linear projection from flattened conv output to output_dim
        self.linear = nn.Linear(conv_output_size, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolutional subsampling to input sequence.

        Args:
            x: Input tensor of shape [batch, input_dim, time].

        Returns:
            Output tensor of shape [batch, time//4, output_dim].
        """
        # Add channel dimension: [batch, 1, input_dim, time]
        x = x.unsqueeze(1)

        # Apply first convolution with ReLU activation
        # [batch, 1, input_dim, time] -> [batch, 32, input_dim//2, time//2]
        x = self.conv1(x)
        x = F.relu(x)

        # Apply second convolution with ReLU activation
        # [batch, 32, input_dim//2, time//2] -> [batch, 32, input_dim//4, time//4]
        x = self.conv2(x)
        x = F.relu(x)

        # Reshape for linear projection
        # [batch, 32, input_dim//4, time//4] -> [batch, time//4, 32 * (input_dim//4)]
        batch_size, channels, freq_bins, time_steps = x.shape
        x = x.transpose(2, 3)  # [batch, 32, time//4, input_dim//4]
        x = x.contiguous().view(batch_size, time_steps, channels * freq_bins)

        # Project to output dimension
        # [batch, time//4, 32 * (input_dim//4)] -> [batch, time//4, output_dim]
        x = self.linear(x)
        x = self.dropout(x)

        return x
