"""Duration categories and utilities for chord voicing generation.

This module defines duration classes that control how long chords sustain,
ranging from quick staccato articulations to slow atmospheric pads.
"""

import random
from typing import Optional


DURATION_OPTIONS = {
    "very_short": {
        "min": 0.5,
        "max": 1.0,
        "description": "Quick staccato chords"
    },
    "short": {
        "min": 1.0,
        "max": 1.5,
        "description": "Brief chord stabs"
    },
    "medium": {
        "min": 1.5,
        "max": 2.5,
        "description": "Standard sustained chords"
    },
    "long": {
        "min": 2.5,
        "max": 4.0,
        "description": "Extended sustain"
    },
    "very_long": {
        "min": 4.0,
        "max": 5.0,
        "description": "Slow, atmospheric pads"
    }
}


def select_duration(duration_class: str, seed: Optional[int] = None) -> float:
    """Select a random duration within a specified duration class range.

    Args:
        duration_class: Key from DURATION_OPTIONS specifying the duration category
            (very_short, short, medium, long, very_long)
        seed: Optional random seed for reproducibility

    Returns:
        A random duration value in seconds within the specified class range

    Raises:
        KeyError: If duration_class is not found in DURATION_OPTIONS

    Examples:
        >>> duration = select_duration("medium")
        >>> 1.5 <= duration <= 2.5
        True

        >>> duration = select_duration("short", seed=42)
        >>> duration  # Will always return same value with same seed
    """
    if seed is not None:
        random.seed(seed)

    if duration_class not in DURATION_OPTIONS:
        raise KeyError(
            f"Duration class '{duration_class}' not found. "
            f"Available options: {list(DURATION_OPTIONS.keys())}"
        )

    duration_config = DURATION_OPTIONS[duration_class]
    return random.uniform(duration_config["min"], duration_config["max"])
