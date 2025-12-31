"""Conformer building blocks for audio processing.

This module implements the Conformer architecture components including positional
encoding, convolution modules, feed-forward modules, multi-head self-attention,
and complete Conformer blocks for sequence-to-sequence audio modeling.

References:
    Gulati, A., et al. (2020). "Conformer: Convolution-augmented Transformer for Speech Recognition"
    https://arxiv.org/abs/2005.08100
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models.

    Encodes the absolute position of tokens in the sequence using sine and cosine
    functions of different frequencies. This allows the model to attend to relative
    positions and distance between tokens.

    Args:
        d_model: The dimensionality of the model embeddings.
        max_len: Maximum sequence length. Defaults to 5000.
        dropout: Dropout probability applied to the positional encodings. Defaults to 0.1.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        """Initialize positional encoding.

        Args:
            d_model: Model dimensionality.
            max_len: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Tensor of shape [batch_size, seq_len, d_model] with positional encoding added.
        """
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        pos_encoding = self.pe[:seq_len, :].unsqueeze(0)  # [1, seq_len, d_model]
        x = x + pos_encoding
        return self.dropout(x)


class ConvolutionModule(nn.Module):
    """Conformer convolution module with depthwise separable convolution.

    Applies a lightweight convolutional layer with gating and swish activation
    to capture local dependencies in the sequence. Uses depthwise separable
    convolution to reduce parameters.

    Structure:
        LayerNorm → Pointwise Conv (2x expansion) → GLU →
        Depthwise Conv (kernel_size) → BatchNorm → Swish →
        Pointwise Conv (1x contraction) → Dropout

    Args:
        d_model: Model dimensionality.
        kernel_size: Kernel size for depthwise convolution. Defaults to 31.
        dropout: Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        """Initialize convolution module.

        Args:
            d_model: Model dimensionality.
            kernel_size: Depthwise convolution kernel size.
            dropout: Dropout rate.
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise convolution to expand (with GLU)
        self.pointwise_conv1 = nn.Conv1d(
            d_model,
            2 * d_model,
            kernel_size=1,
            padding=0,
        )

        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)

        # Pointwise convolution to contract
        self.pointwise_conv2 = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=1,
            padding=0,
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution module to input.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Tensor of shape [batch_size, seq_len, d_model].
        """
        # Layer norm
        x = self.layer_norm(x)  # [batch, seq_len, d_model]

        # Transpose for conv1d: [batch, d_model, seq_len]
        x = x.transpose(1, 2)

        # Pointwise conv with GLU activation
        x = self.pointwise_conv1(x)  # [batch, 2*d_model, seq_len]
        x, gate = x.chunk(2, dim=1)  # Two chunks of [batch, d_model, seq_len]
        x = x * torch.sigmoid(gate)  # GLU: multiply by sigmoid(gate)

        # Depthwise convolution
        x = self.depthwise_conv(x)  # [batch, d_model, seq_len]
        x = self.batch_norm(x)
        x = F.silu(x)  # Swish activation

        # Pointwise convolution (contract)
        x = self.pointwise_conv2(x)  # [batch, d_model, seq_len]
        x = self.dropout(x)

        # Transpose back: [batch, seq_len, d_model]
        x = x.transpose(1, 2)

        return x


class FeedForwardModule(nn.Module):
    """Feed-forward module with two linear layers and Swish activation.

    A simple feed-forward network with expansion and contraction of dimensions.
    Used in Conformer blocks with macaron-style half-weighted versions.

    Structure:
        LayerNorm → Linear (expansion) → Swish → Dropout → Linear (contraction) → Dropout

    Args:
        d_model: Model dimensionality.
        expansion_factor: Expansion factor for hidden dimension. Defaults to 4.
        dropout: Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize feed-forward module.

        Args:
            d_model: Model dimensionality.
            expansion_factor: Hidden dimension expansion factor.
            dropout: Dropout rate.
        """
        super().__init__()

        hidden_dim = d_model * expansion_factor

        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward module to input.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Tensor of shape [batch_size, seq_len, d_model].
        """
        # Layer norm
        x = self.layer_norm(x)  # [batch, seq_len, d_model]

        # First linear layer with Swish activation
        x = self.linear1(x)  # [batch, seq_len, hidden_dim]
        x = F.silu(x)  # Swish activation
        x = self.dropout1(x)

        # Second linear layer
        x = self.linear2(x)  # [batch, seq_len, d_model]
        x = self.dropout2(x)

        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with pre-normalization.

    Standard multi-head scaled dot-product attention used in transformer models.
    Pre-normalization is applied before the attention operation.

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize multi-head self-attention.

        Args:
            d_model: Model dimensionality.
            num_heads: Number of attention heads.
            dropout: Dropout rate.

        Raises:
            ValueError: If d_model is not divisible by num_heads.
        """
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.layer_norm = nn.LayerNorm(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply multi-head self-attention.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].
            mask: Optional attention mask of shape [batch_size, seq_len] or
                [batch_size, seq_len, seq_len]. True values indicate positions to mask.
                Defaults to None.

        Returns:
            Tensor of shape [batch_size, seq_len, d_model].
        """
        batch_size, seq_len, _ = x.shape

        # Pre-normalization
        x_norm = self.layer_norm(x)  # [batch, seq_len, d_model]

        # Project to Q, K, V
        query = self.query_proj(x_norm)  # [batch, seq_len, d_model]
        key = self.key_proj(x_norm)      # [batch, seq_len, d_model]
        value = self.value_proj(x_norm)  # [batch, seq_len, d_model]

        # Reshape for multi-head attention
        # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, head_dim]
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores shape: [batch, num_heads, seq_len, seq_len]

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # [batch, seq_len] -> [batch, 1, 1, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                # [batch, seq_len, seq_len] -> [batch, 1, seq_len, seq_len]
                mask = mask.unsqueeze(1)

            # Mask is True where we want to attend, so we set False positions to -inf
            scores = scores.masked_fill(~mask, float("-inf"))

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        # attn_output shape: [batch, num_heads, seq_len, head_dim]

        # Transpose and reshape back
        attn_output = attn_output.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.output_proj(attn_output)  # [batch, seq_len, d_model]

        return output


class ConformerBlock(nn.Module):
    """Full Conformer block with macaron-style feed-forward networks.

    The Conformer block combines multiple sub-modules for effective sequence modeling:
    - Feed-forward modules (with macaron-style half-weighting) for non-linear transformation
    - Multi-head self-attention for modeling long-range dependencies
    - Convolution module for capturing local context

    Structure (with residual connections):
        FFN(½) → MHSA → Conv → FFN(½) → LayerNorm

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads. Defaults to 8.
        conv_kernel_size: Kernel size for convolution module. Defaults to 31.
        ff_expansion_factor: Expansion factor for feed-forward hidden dimension. Defaults to 4.
        dropout: Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        conv_kernel_size: int = 31,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize Conformer block.

        Args:
            d_model: Model dimensionality.
            num_heads: Number of attention heads.
            conv_kernel_size: Convolution kernel size.
            ff_expansion_factor: Feed-forward expansion factor.
            dropout: Dropout rate.
        """
        super().__init__()

        # Feed-forward modules (macaron-style with 0.5 weighting)
        self.ff1 = FeedForwardModule(
            d_model=d_model,
            expansion_factor=ff_expansion_factor,
            dropout=dropout,
        )

        # Multi-head self-attention
        self.mhsa = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Convolution module
        self.conv = ConvolutionModule(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        # Second feed-forward module
        self.ff2 = FeedForwardModule(
            d_model=d_model,
            expansion_factor=ff_expansion_factor,
            dropout=dropout,
        )

        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply Conformer block to input with residual connections.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].
            mask: Optional attention mask. See MultiHeadSelfAttention.forward for details.
                Defaults to None.

        Returns:
            Tensor of shape [batch_size, seq_len, d_model].
        """
        # Macaron-style feed-forward with 0.5 weighting
        ff1_out = self.ff1(x)
        x = x + 0.5 * ff1_out

        # Multi-head self-attention
        mhsa_out = self.mhsa(x, mask=mask)
        x = x + mhsa_out

        # Convolution module
        conv_out = self.conv(x)
        x = x + conv_out

        # Second feed-forward with 0.5 weighting
        ff2_out = self.ff2(x)
        x = x + 0.5 * ff2_out

        # Final layer normalization
        x = self.final_layer_norm(x)

        return x
