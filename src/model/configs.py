"""Preset model configurations for ChordAI.

This module provides predefined configurations for different model sizes,
ranging from small models suitable for testing to large models for production.
"""

from .chord_model import ChordModelConfig


# Small configuration for testing and debugging
SMALL_CONFIG = ChordModelConfig(
    d_model=128,
    num_conformer_blocks=4,
    num_attention_heads=4,
    conv_kernel_size=15,
    ff_expansion_factor=2,
    dropout=0.1,
)

# Medium configuration for initial training
MEDIUM_CONFIG = ChordModelConfig(
    d_model=256,
    num_conformer_blocks=6,
    num_attention_heads=8,
    conv_kernel_size=31,
    ff_expansion_factor=4,
    dropout=0.1,
)

# Large configuration for best performance
LARGE_CONFIG = ChordModelConfig(
    d_model=384,
    num_conformer_blocks=8,
    num_attention_heads=8,
    conv_kernel_size=31,
    ff_expansion_factor=4,
    dropout=0.15,
)


def get_config(name: str) -> ChordModelConfig:
    """Get a preset model configuration by name.

    Args:
        name: The name of the configuration. Valid options are:
            - 'small': Lightweight model for testing/debugging
            - 'medium': Medium-sized model for initial training
            - 'large': Large model for best performance

    Returns:
        ChordModelConfig: The requested configuration object.

    Raises:
        ValueError: If the configuration name is not recognized.

    Example:
        >>> config = get_config('medium')
        >>> print(config.d_model)
        256
    """
    configs = {
        'small': SMALL_CONFIG,
        'medium': MEDIUM_CONFIG,
        'large': LARGE_CONFIG,
    }

    if name.lower() not in configs:
        valid_names = ', '.join(sorted(configs.keys()))
        raise ValueError(
            f"Unknown configuration name: '{name}'. "
            f"Valid options are: {valid_names}"
        )

    return configs[name.lower()]
