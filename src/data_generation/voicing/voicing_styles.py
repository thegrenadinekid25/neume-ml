"""Voicing styles and utilities for distributing chord voices across a range.

This module defines different voicing styles that control how pitch classes
are distributed across voice parts, from tight closed voicings to wide open
spreads across the full instrumental range.
"""

from typing import List, Tuple
import random


VOICING_STYLES = {
    "closed": {
        "description": "All voices within one octave",
        "max_span": 12,
        "min_spacing": 1,
        "prefer_clusters": True
    },
    "open": {
        "description": "Traditional SATB spacing with gaps",
        "max_span": 36,
        "min_spacing": 2,
        "prefer_clusters": False
    },
    "mixed": {
        "description": "Combination of open and closed positions",
        "max_span": 24,
        "min_spacing": 2,
        "prefer_clusters": False
    },
    "wide": {
        "description": "Maximum spread across full range",
        "max_span": 48,
        "min_spacing": 3,
        "prefer_clusters": False
    }
}


def apply_voicing_style(
    base_pitches: List[int],
    num_voices: int,
    style: str,
    voice_ranges: List[Tuple[int, int]]
) -> List[int]:
    """Distribute base pitch classes across voices according to voicing style.

    Takes a set of pitch classes (0-11) and distributes them across multiple
    voice parts, following the constraints and preferences of the specified
    voicing style.

    Args:
        base_pitches: List of pitch classes to voice, values in range [0, 11].
            Will be cycled if fewer pitches than voices.
        num_voices: Number of voice parts to generate, typically 4-16.
        style: Name of voicing style to apply. Must be key in VOICING_STYLES
            (closed, open, mixed, wide).
        voice_ranges: List of (low, high) MIDI note ranges for each voice.
            Length should match num_voices. Each tuple specifies the valid
            MIDI range for that voice part.

    Returns:
        List of MIDI note numbers, one per voice, in ascending order.

    Raises:
        KeyError: If style is not found in VOICING_STYLES.
        ValueError: If voice_ranges length doesn't match num_voices.
        ValueError: If base_pitches is empty.

    Examples:
        >>> base_pitches = [0, 4, 7]  # C, E, G
        >>> voice_ranges = [(36, 48), (48, 60), (60, 72), (72, 84)]
        >>> pitches = apply_voicing_style(base_pitches, 4, "closed", voice_ranges)
        >>> len(pitches) == 4
        True
        >>> pitches == sorted(pitches)  # Returned in ascending order
        True
    """
    if not base_pitches:
        raise ValueError("base_pitches cannot be empty")

    if style not in VOICING_STYLES:
        raise KeyError(
            f"Voicing style '{style}' not found. "
            f"Available options: {list(VOICING_STYLES.keys())}"
        )

    if len(voice_ranges) != num_voices:
        raise ValueError(
            f"voice_ranges length ({len(voice_ranges)}) must match "
            f"num_voices ({num_voices})"
        )

    style_config = VOICING_STYLES[style]
    max_span = style_config["max_span"]
    min_spacing = style_config["min_spacing"]
    prefer_clusters = style_config["prefer_clusters"]

    voicing = []

    if style == "closed":
        voicing = _apply_closed_voicing(
            base_pitches, num_voices, voice_ranges, max_span, min_spacing
        )
    elif style == "open":
        voicing = _apply_open_voicing(
            base_pitches, num_voices, voice_ranges, max_span, min_spacing
        )
    elif style == "mixed":
        voicing = _apply_mixed_voicing(
            base_pitches, num_voices, voice_ranges, max_span, min_spacing
        )
    elif style == "wide":
        voicing = _apply_wide_voicing(
            base_pitches, num_voices, voice_ranges, max_span, min_spacing
        )

    return sorted(voicing)


def _apply_closed_voicing(
    base_pitches: List[int],
    num_voices: int,
    voice_ranges: List[Tuple[int, int]],
    max_span: int,
    min_spacing: int
) -> List[int]:
    """Pack voices tightly within max_span, preferring clusters.

    Tries to keep all voices within a single octave (or max_span semitones),
    grouping note repetitions and close intervals together.
    """
    voicing = []
    pitch_cycle = (base_pitches * ((num_voices // len(base_pitches)) + 1))[:num_voices]

    # Find the lowest available starting point
    min_note = min(low for low, _ in voice_ranges)
    current_note = min_note

    for i, pitch_class in enumerate(pitch_cycle):
        voice_low, voice_high = voice_ranges[i]

        # Adjust pitch class to fit within voice range and span constraint
        candidate = current_note + (pitch_class - (current_note % 12))
        if candidate < voice_low:
            candidate += 12
        if candidate > voice_high:
            candidate -= 12

        # Ensure we stay within max_span from first note
        if i > 0 and (candidate - voicing[0]) > max_span:
            candidate = min(candidate, voicing[0] + max_span)

        # Enforce minimum spacing
        if i > 0 and (candidate - voicing[-1]) < min_spacing:
            candidate = voicing[-1] + min_spacing

        # Final boundary check
        candidate = max(voice_low, min(candidate, voice_high))

        voicing.append(candidate)
        current_note = candidate

    return voicing


def _apply_open_voicing(
    base_pitches: List[int],
    num_voices: int,
    voice_ranges: List[Tuple[int, int]],
    max_span: int,
    min_spacing: int
) -> List[int]:
    """Spread voices with gaps, traditional SATB-like spacing.

    Distributes voices across the available range with intentional gaps,
    avoiding tight clustering. Each voice tends toward the middle of its range.
    """
    voicing = []
    pitch_cycle = (base_pitches * ((num_voices // len(base_pitches)) + 1))[:num_voices]

    for i, pitch_class in enumerate(pitch_cycle):
        voice_low, voice_high = voice_ranges[i]
        voice_range = voice_high - voice_low

        # Place voice toward middle of its range
        mid_point = voice_low + (voice_range // 2)
        candidate = mid_point - (mid_point % 12) + pitch_class

        if candidate < voice_low:
            candidate += 12
        if candidate > voice_high:
            candidate -= 12

        # Enforce minimum spacing from previous voice
        if i > 0 and (candidate - voicing[-1]) < min_spacing:
            candidate = voicing[-1] + min_spacing

        # Final boundary check
        candidate = max(voice_low, min(candidate, voice_high))

        voicing.append(candidate)

    return voicing


def _apply_mixed_voicing(
    base_pitches: List[int],
    num_voices: int,
    voice_ranges: List[Tuple[int, int]],
    max_span: int,
    min_spacing: int
) -> List[int]:
    """Combination of open and closed positions.

    Alternates between clustered sections and spaced sections to create
    variety while maintaining coherence.
    """
    voicing = []
    pitch_cycle = (base_pitches * ((num_voices // len(base_pitches)) + 1))[:num_voices]

    # Divide voices into alternating closed and open sections
    current_note = min(low for low, _ in voice_ranges)

    for i, pitch_class in enumerate(pitch_cycle):
        voice_low, voice_high = voice_ranges[i]

        # Alternate between tight and spread behavior
        is_tight_section = (i // 2) % 2 == 0

        candidate = current_note + (pitch_class - (current_note % 12))
        if candidate < voice_low:
            candidate += 12
        if candidate > voice_high:
            candidate -= 12

        # Apply spacing rules based on section
        if i > 0:
            spacing = candidate - voicing[-1]
            if is_tight_section and spacing < min_spacing:
                candidate = voicing[-1] + min_spacing
            elif not is_tight_section and spacing < min_spacing + 2:
                candidate = voicing[-1] + min_spacing + 2

        candidate = max(voice_low, min(candidate, voice_high))
        voicing.append(candidate)
        current_note = candidate

    return voicing


def _apply_wide_voicing(
    base_pitches: List[int],
    num_voices: int,
    voice_ranges: List[Tuple[int, int]],
    max_span: int,
    min_spacing: int
) -> List[int]:
    """Maximum spread across full available range.

    Distributes voices to span the widest possible range, placing each voice
    at extremes within its range to maximize separation.
    """
    voicing = []
    pitch_cycle = (base_pitches * ((num_voices // len(base_pitches)) + 1))[:num_voices]

    for i, pitch_class in enumerate(pitch_cycle):
        voice_low, voice_high = voice_ranges[i]

        # For lower voices, place near bottom; for upper voices, near top
        if i < num_voices // 2:
            # Lower voices: prefer lower part of range
            candidate = voice_low + pitch_class
            if candidate > voice_low + 6:
                candidate -= 12
        else:
            # Upper voices: prefer upper part of range
            candidate = voice_high - (12 - pitch_class)
            if candidate < voice_high - 6:
                candidate += 12

        # Enforce minimum spacing
        if i > 0 and (candidate - voicing[-1]) < min_spacing:
            candidate = voicing[-1] + min_spacing

        # Final boundary check
        candidate = max(voice_low, min(candidate, voice_high))

        voicing.append(candidate)

    return voicing
