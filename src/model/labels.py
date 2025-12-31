"""Chord type labels and pitch class mappings for model training.

This module provides chord type and pitch class labels, bidirectional index mappings,
and utilities for formatting human-readable chord names. It serves as a centralized
reference for all chord vocabulary used in the model.
"""

from typing import Dict

from src.data_generation.voicing.chord_types import CHORD_TYPES, ROOT_NAMES


# Bidirectional chord type mappings (38 chord types)
CHORD_TYPE_TO_IDX: Dict[str, int] = {chord_type: idx for idx, chord_type in enumerate(sorted(CHORD_TYPES.keys()))}
IDX_TO_CHORD_TYPE: Dict[int, str] = {idx: chord_type for chord_type, idx in CHORD_TYPE_TO_IDX.items()}

# Pitch class mappings (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
PITCH_CLASS_TO_IDX: Dict[str, int] = {pitch: idx for idx, pitch in enumerate(ROOT_NAMES)}
IDX_TO_PITCH_CLASS: Dict[int, str] = {idx: pitch for pitch, idx in PITCH_CLASS_TO_IDX.items()}


def format_chord_name(
    chord_type_idx: int,
    root_idx: int,
    bass_idx: int | None = None,
) -> str:
    """
    Format chord indices into human-readable chord name.

    Converts numerical indices for chord type, root note, and optional bass note
    into a standard chord naming notation (e.g., "C major", "Am7/E").

    Args:
        chord_type_idx: Index into IDX_TO_CHORD_TYPE (0-37 for 38 chord types).
        root_idx: Index into IDX_TO_PITCH_CLASS (0-11 for 12 pitch classes).
        bass_idx: Optional index into IDX_TO_PITCH_CLASS for slash chord bass note.
                 If None, no slash notation is added.

    Returns:
        Formatted chord name string (e.g., "C major", "Am7", "G major/B").

    Raises:
        ValueError: If chord_type_idx is outside 0-37 range.
        ValueError: If root_idx is outside 0-11 range.
        ValueError: If bass_idx is provided but outside 0-11 range.
        KeyError: If indices don't map to valid chord types or pitch classes.

    Example:
        >>> format_chord_name(26, 0)  # Major chord on C (if major is at index 26)
        'C major'
        >>> format_chord_name(11, 7)  # dom7 on G (if dom7 is at index 11)
        'G dom7'
        >>> format_chord_name(26, 7, 11)  # G major with B in bass
        'G major/B'
    """
    # Validate indices
    num_chord_types = len(CHORD_TYPE_TO_IDX)
    if not 0 <= chord_type_idx < num_chord_types:
        raise ValueError(f"chord_type_idx must be in range 0-{num_chord_types - 1}, got {chord_type_idx}")
    if not 0 <= root_idx <= 11:
        raise ValueError(f"root_idx must be in range 0-11, got {root_idx}")
    if bass_idx is not None and not 0 <= bass_idx <= 11:
        raise ValueError(f"bass_idx must be in range 0-11 or None, got {bass_idx}")

    # Get chord type and root names
    chord_type = IDX_TO_CHORD_TYPE[chord_type_idx]
    root = IDX_TO_PITCH_CLASS[root_idx]

    # Build base chord name
    chord_name = f"{root} {chord_type}"

    # Add slash notation for bass note if provided
    if bass_idx is not None:
        bass_note = IDX_TO_PITCH_CLASS[bass_idx]
        chord_name = f"{chord_name}/{bass_note}"

    return chord_name


def get_num_chord_types() -> int:
    """
    Get the total number of chord types.

    Returns:
        Number of unique chord types in the model vocabulary (38).
    """
    return len(CHORD_TYPE_TO_IDX)


def get_num_pitch_classes() -> int:
    """
    Get the total number of pitch classes.

    Returns:
        Number of unique pitch classes (12).
    """
    return len(PITCH_CLASS_TO_IDX)
