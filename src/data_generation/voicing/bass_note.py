"""
Module for handling bass note selection for chord inversions and slash chords.

This module provides functionality to select appropriate bass notes based on
chord type, inversion, and voicing preferences.
"""

from typing import List
import random


BASS_NOTE_TYPES = {
    "root": "Bass is the root of the chord",
    "first_inv": "Bass is the 3rd of the chord",
    "second_inv": "Bass is the 5th of the chord",
    "third_inv": "Bass is the 7th of the chord (only for 7th chords)",
    "slash": "Bass is a non-chord-tone from the scale",
}


def get_inversion_interval(chord_intervals: List[int], inversion_type: str) -> int:
    """
    Get the interval for a given inversion type from the chord intervals.

    Args:
        chord_intervals: List of intervals in semitones from the root (e.g., [0, 4, 7] for major triad)
        inversion_type: Type of inversion ("root", "first_inv", "second_inv", "third_inv")

    Returns:
        The interval in semitones for the specified inversion type

    Raises:
        ValueError: If the inversion type is not valid or the chord doesn't have the required interval
    """
    if inversion_type == "root":
        return 0

    elif inversion_type == "first_inv":
        # Find the third (intervals 3 or 4)
        for interval in chord_intervals:
            if interval in (3, 4):
                return interval
        raise ValueError(f"No third found in chord intervals: {chord_intervals}")

    elif inversion_type == "second_inv":
        # Find the fifth (intervals 6, 7, or 8)
        for interval in chord_intervals:
            if interval in (6, 7, 8):
                return interval
        raise ValueError(f"No fifth found in chord intervals: {chord_intervals}")

    elif inversion_type == "third_inv":
        # Find the seventh (intervals 9, 10, or 11)
        for interval in chord_intervals:
            if interval in (9, 10, 11):
                return interval
        raise ValueError(f"No seventh found in chord intervals: {chord_intervals}")

    else:
        raise ValueError(f"Unknown inversion type: {inversion_type}")


def select_bass_note(
    chord_type: str,
    root: int,
    inversion_type: str,
    chord_intervals: List[int]
) -> int:
    """
    Select the bass note pitch class for a chord based on inversion type.

    Args:
        chord_type: Type of chord ("major", "minor", "dominant", etc.)
        root: Root pitch class of the chord (0-11)
        inversion_type: Type of inversion or voicing ("root", "first_inv", "second_inv", "third_inv", "slash")
        chord_intervals: List of intervals in semitones from the root

    Returns:
        The pitch class (0-11) of the bass note

    Raises:
        ValueError: If inversion type is invalid or chord lacks required intervals for the inversion
    """
    if inversion_type == "slash":
        return _select_slash_bass_note(chord_type, root, chord_intervals)

    # For root, first_inv, second_inv, third_inv
    if inversion_type == "third_inv":
        # Check if chord has a 7th (4+ notes with a 7th interval)
        has_seventh = any(interval in (9, 10, 11) for interval in chord_intervals)
        if not has_seventh:
            # Fall back to root position if no 7th is present
            return root % 12

    try:
        interval = get_inversion_interval(chord_intervals, inversion_type)
        return (root + interval) % 12
    except ValueError:
        # Fall back to root position if the required interval is not found
        return root % 12


def _select_slash_bass_note(chord_type: str, root: int, chord_intervals: List[int]) -> int:
    """
    Select a random slash bass note from scale tones not in the chord.

    Args:
        chord_type: Type of chord ("major", "minor", "dominant", etc.)
        root: Root pitch class of the chord (0-11)
        chord_intervals: List of intervals in semitones from the root

    Returns:
        The pitch class (0-11) of a valid slash bass note
    """
    # Determine scale based on chord type
    if "minor" in chord_type.lower():
        # Natural minor scale: 0, 2, 3, 5, 7, 8, 10
        scale = [0, 2, 3, 5, 7, 8, 10]
    else:
        # Major scale: 0, 2, 4, 5, 7, 9, 11
        scale = [0, 2, 4, 5, 7, 9, 11]

    # Get pitch classes of chord tones
    chord_tones = {(root + interval) % 12 for interval in chord_intervals}

    # Find scale tones that are not in the chord
    valid_slash_notes = [
        (root + scale_degree) % 12
        for scale_degree in scale
        if (root + scale_degree) % 12 not in chord_tones
    ]

    # If no valid slash notes exist, return a random scale tone
    if not valid_slash_notes:
        valid_slash_notes = [(root + scale_degree) % 12 for scale_degree in scale]

    return random.choice(valid_slash_notes)
