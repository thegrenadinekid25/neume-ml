"""Shell voicing strategies for selecting which chord tones to include.

This module defines strategies for choosing a subset of chord intervals based on
voicing principles (e.g., jazz shell voicings, rootless voicings, quartal harmony).
Each strategy prioritizes different interval categories to achieve a specific sound.
"""

from typing import List, Dict, Any
from .chord_types import categorize_intervals


# Define the 5 shell voicing strategies
SHELL_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "classical": {
        "description": "Root and fifth emphasized, full triadic sound",
        "priority": ["root", "fifth", "third", "seventh", "extensions"],
    },
    "jazz_shell": {
        "description": "3rd and 7th define quality, root optional, omit 5th",
        "priority": ["third", "seventh", "root", "extensions", "fifth"],
    },
    "rootless_a": {
        "description": "Type A rootless voicing, assume bass plays root",
        "priority": ["third", "seventh", "extensions", "fifth", "root"],
    },
    "rootless_b": {
        "description": "Type B rootless voicing, inverted from Type A",
        "priority": ["seventh", "third", "extensions", "fifth", "root"],
    },
    "quartal": {
        "description": "Stack fourths, avoid thirds for modern sound",
        "priority": ["extensions", "seventh", "root", "fifth", "third"],
    },
}


def select_pitches(
    chord_intervals: List[int], num_pitches: int, strategy: str
) -> List[int]:
    """Select which intervals to include based on strategy priority.

    Categorizes the available chord intervals and selects a subset based on
    the priority ordering defined in the specified strategy. Intervals are
    selected to match the strategy's priority categories until the desired
    number of pitches is reached.

    Args:
        chord_intervals: All intervals in the chord (e.g., [0, 4, 7, 10]).
            Expected to be semi-tones relative to root (0 = root).
        num_pitches: How many distinct pitches to include. Must be between
            1 and len(chord_intervals).
        strategy: Name of strategy to use. Must be a key in SHELL_STRATEGIES.

    Returns:
        List of selected intervals, sorted in ascending order.

    Raises:
        ValueError: If strategy name is not found or num_pitches is invalid.

    Example:
        >>> intervals = [0, 4, 7, 10]  # Root, 3rd, 5th, 7th
        >>> select_pitches(intervals, 3, "jazz_shell")
        [0, 4, 10]  # 3rd, 7th, root
    """
    if strategy not in SHELL_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Available strategies: {list(SHELL_STRATEGIES.keys())}"
        )

    if num_pitches < 1 or num_pitches > len(chord_intervals):
        raise ValueError(
            f"num_pitches must be between 1 and {len(chord_intervals)}, "
            f"got {num_pitches}"
        )

    # Categorize the available intervals
    categorized = categorize_intervals(chord_intervals)

    # Get the priority list for this strategy
    priority = SHELL_STRATEGIES[strategy]["priority"]

    # Select intervals following the priority order
    selected: List[int] = []

    for category in priority:
        if category in categorized:
            for interval in categorized[category]:
                if len(selected) < num_pitches:
                    selected.append(interval)
                else:
                    break

        if len(selected) >= num_pitches:
            break

    # Return sorted list
    return sorted(selected)
