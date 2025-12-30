"""
Module defining chord types as interval lists from the root.

This module provides a comprehensive collection of 32 chord types used in music
generation and analysis, with utilities for working with chord intervals and
pitch classes.
"""

from typing import Dict, List


# 32 Chord types mapped by name to interval lists
CHORD_TYPES: Dict[str, List[int]] = {
    # Triads (4)
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "diminished": [0, 3, 6],
    "augmented": [0, 4, 8],

    # 7ths (6)
    "maj7": [0, 4, 7, 11],
    "dom7": [0, 4, 7, 10],
    "min7": [0, 3, 7, 10],
    "minmaj7": [0, 3, 7, 11],
    "half_dim7": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],

    # Suspended (2)
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],

    # Added (5)
    "add9": [0, 4, 7, 14],
    "add11": [0, 4, 7, 17],
    "madd9": [0, 3, 7, 14],
    "6": [0, 4, 7, 9],
    "m6": [0, 3, 7, 9],

    # Extended (7)
    "maj9": [0, 4, 7, 11, 14],
    "dom9": [0, 4, 7, 10, 14],
    "min9": [0, 3, 7, 10, 14],
    "dom11": [0, 4, 7, 10, 14, 17],
    "min11": [0, 3, 7, 10, 14, 17],
    "dom13": [0, 4, 7, 10, 14, 17, 21],
    "maj13": [0, 4, 7, 11, 14, 17, 21],

    # Altered (4)
    "dom7b9": [0, 4, 7, 10, 13],
    "dom7sharp9": [0, 4, 7, 10, 15],
    "dom7b5": [0, 4, 6, 10],
    "dom7sharp5": [0, 4, 8, 10],

    # Special (4)
    "quartal3": [0, 5, 10],
    "quartal4": [0, 5, 10, 15],
    "cluster_whole": [0, 2, 4],
    "cluster_chromatic": [0, 1, 2],
}

# Root note names (chromatic scale)
ROOT_NAMES: List[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def categorize_intervals(intervals: List[int]) -> Dict[str, List[int]]:
    """
    Categorize intervals into chord components.

    Organizes intervals from a root note into functional categories based on
    their distance from the root, useful for analyzing chord structure.

    Args:
        intervals: List of interval distances in semitones from the root.

    Returns:
        Dictionary with keys:
            - "root": Always contains [0]
            - "third": Intervals 3 or 4 (minor or major third)
            - "fifth": Intervals 6, 7, or 8 (diminished, perfect, or augmented fifth)
            - "seventh": Intervals 9, 10, or 11 (diminished, dominant, or major seventh)
            - "extensions": Any interval >= 12 (9ths, 11ths, 13ths, etc.)

    Example:
        >>> categorize_intervals([0, 4, 7])
        {'root': [0], 'third': [4], 'fifth': [7], 'seventh': [], 'extensions': []}
    """
    categorized = {
        "root": [],
        "third": [],
        "fifth": [],
        "seventh": [],
        "extensions": [],
    }

    for interval in intervals:
        if interval == 0:
            categorized["root"].append(interval)
        elif interval in [3, 4]:
            categorized["third"].append(interval)
        elif interval in [6, 7, 8]:
            categorized["fifth"].append(interval)
        elif interval in [9, 10, 11]:
            categorized["seventh"].append(interval)
        elif interval >= 12:
            categorized["extensions"].append(interval)

    return categorized


def get_pitch_classes(chord_type: str, root: int) -> List[int]:
    """
    Get pitch classes (0-11) for a chord type with a given root note.

    Generates the actual pitch class representation of a chord by adding
    the root note to each interval in the chord type definition, then
    reducing to the 0-11 range using modulo 12 arithmetic.

    Args:
        chord_type: Name of the chord type (e.g., "major", "minor", "dom7").
                   Must be a key in CHORD_TYPES dictionary.
        root: Root note as MIDI pitch class (0-11, where 0 is C, 1 is C#, etc.).
              Values outside 0-11 will be reduced modulo 12.

    Returns:
        List of pitch classes (0-11) representing the chord pitches.

    Raises:
        KeyError: If chord_type is not found in CHORD_TYPES.

    Example:
        >>> get_pitch_classes("major", 0)
        [0, 4, 7]
        >>> get_pitch_classes("major", 2)  # D major
        [2, 6, 9]
        >>> get_pitch_classes("dom7", 5)   # F dominant 7
        [5, 9, 0, 3]
    """
    if chord_type not in CHORD_TYPES:
        raise KeyError(f"Chord type '{chord_type}' not found in CHORD_TYPES")

    intervals = CHORD_TYPES[chord_type]
    root = root % 12  # Ensure root is in valid pitch class range

    pitch_classes = [(root + interval) % 12 for interval in intervals]

    return pitch_classes
