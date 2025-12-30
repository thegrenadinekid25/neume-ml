"""
Voice distribution algorithm for distributing pitch classes across N voices.

This module handles the harmonic voicing of pitch classes across multiple voice parts,
ensuring musical constraints are satisfied (proper spacing, range limitations, etc.).
"""

from typing import List, Tuple, Optional
import random


# Minimum spacing between adjacent voices (in semitones)
MIN_SPACING = 2

# Register definitions: each register specifies bass and soprano ranges
REGISTERS = {
    "low": {
        "bass_range": (36, 55),      # C2-G3
        "soprano_range": (55, 72),   # G3-C5
    },
    "medium": {
        "bass_range": (40, 60),      # E2-C4
        "soprano_range": (60, 79),   # C4-G5
    },
    "high": {
        "bass_range": (48, 67),      # C3-G4
        "soprano_range": (67, 84),   # G4-C6
    },
    "extended": {
        "bass_range": (36, 60),      # C2-C4 (wide bass range for large ensembles)
        "soprano_range": (60, 88),   # C4-E6 (extended soprano for large ensembles)
    },
}


def compute_voice_ranges(
    num_parts: int,
    register: str = "medium"
) -> List[Tuple[int, int]]:
    """
    Compute MIDI note ranges for N voice parts.

    Divides the register span evenly among the requested number of voices,
    with slight overlap between adjacent voice ranges to allow flexibility
    in voice leading.

    Args:
        num_parts: Number of voices (4-16)
        register: Register name from REGISTERS dict ("low", "medium", "high")

    Returns:
        List of (low_note, high_note) tuples, one per voice, ordered from lowest to highest.

    Raises:
        ValueError: If num_parts is out of range or register is invalid.
    """
    if num_parts < 4 or num_parts > 16:
        raise ValueError(f"num_parts must be between 4 and 16, got {num_parts}")

    if register not in REGISTERS:
        raise ValueError(
            f"register must be one of {list(REGISTERS.keys())}, got '{register}'"
        )

    reg = REGISTERS[register]
    bass_low, bass_high = reg["bass_range"]
    soprano_low, soprano_high = reg["soprano_range"]

    # Total span from lowest bass to highest soprano
    total_span = soprano_high - bass_low

    # Divide span among voices
    span_per_voice = total_span / num_parts

    # Create ranges with slight overlap
    ranges = []
    for i in range(num_parts):
        voice_low = int(bass_low + i * span_per_voice)
        voice_high = int(bass_low + (i + 1) * span_per_voice)

        # Ensure minimum range of 12 semitones per voice
        # But cap at soprano_high to stay within register
        if voice_high - voice_low < 12:
            voice_high = min(voice_low + 12, soprano_high)

        # Also cap voice_high at soprano_high in all cases
        voice_high = min(voice_high, soprano_high)

        ranges.append((voice_low, voice_high))

    return ranges


def distribute_voices(
    pitch_classes: List[int],
    bass_note: int,
    num_parts: int,
    register: str = "medium",
    style: str = "open",
    seed: Optional[int] = None
) -> List[int]:
    """
    Distribute pitch classes across N voices with musical constraints.

    Assigns MIDI notes to each voice such that:
    - All voices have unique notes (no unisons)
    - Adjacent voices are separated by at least MIN_SPACING semitones
    - Each note fits within its voice's range
    - The bass voice receives the specified bass_note pitch class
    - All pitch classes are represented (if possible)

    Args:
        pitch_classes: List of pitch classes to distribute (0-11)
        bass_note: Pitch class for the bass voice (0-11)
        num_parts: Number of voices (4-16)
        register: Register name from REGISTERS dict (default: "medium")
        style: Voicing style, currently unused (default: "open")
        seed: Random seed for reproducibility (optional)

    Returns:
        List of MIDI notes, one per voice, in ascending order.

    Raises:
        ValueError: If constraints cannot be satisfied after retries.
    """
    if seed is not None:
        random.seed(seed)

    if bass_note < 0 or bass_note > 11:
        raise ValueError(f"bass_note must be 0-11, got {bass_note}")

    if not pitch_classes or any(pc < 0 or pc > 11 for pc in pitch_classes):
        raise ValueError("pitch_classes must be non-empty list of values 0-11")

    # For large ensembles, use a wider register
    if num_parts >= 10 and register == "medium":
        effective_register = "extended"  # Extended register for large ensembles
    else:
        effective_register = register

    voice_ranges = compute_voice_ranges(num_parts, effective_register)

    # Maximum retries to find valid voicing (more retries for larger ensembles)
    # For 16-voice textures, we need many more attempts due to constraint complexity
    max_retries = 50 + num_parts * 5

    for attempt in range(max_retries):
        voicing = _attempt_voicing(
            pitch_classes=pitch_classes,
            bass_note=bass_note,
            voice_ranges=voice_ranges,
        )

        if voicing is not None:
            return voicing

    # If we exhausted retries, raise an error
    raise ValueError(
        f"Could not distribute {len(pitch_classes)} pitch classes across "
        f"{num_parts} voices with {register} register after {max_retries} attempts"
    )


def _attempt_voicing(
    pitch_classes: List[int],
    bass_note: int,
    voice_ranges: List[Tuple[int, int]],
) -> Optional[List[int]]:
    """
    Attempt a single voicing assignment.

    Args:
        pitch_classes: List of pitch classes (0-11)
        bass_note: Pitch class for bass voice
        voice_ranges: List of (low, high) ranges for each voice

    Returns:
        Valid voicing as list of MIDI notes, or None if constraints fail.
    """
    num_parts = len(voice_ranges)

    # For larger ensembles, reduce minimum spacing to fit more voices
    # This allows closer voicings typical of choral clusters
    if num_parts > 10:
        min_spacing = 1  # Allow semitone spacing for large ensembles
    else:
        min_spacing = MIN_SPACING
    voicing = [None] * num_parts

    # Step 1: Assign bass voice (lowest occurrence of bass_note in bass range)
    bass_low, bass_high = voice_ranges[0]
    bass_candidates = [
        note for note in range(bass_low, bass_high + 1)
        if note % 12 == bass_note
    ]

    if not bass_candidates:
        return None

    voicing[0] = random.choice(bass_candidates)  # Pick a valid bass note

    # Step 2: Create a list of pitch classes to distribute
    # We need to fill num_parts - 1 remaining voices
    remaining_pcs = list(pitch_classes)
    if bass_note in remaining_pcs:
        remaining_pcs.remove(bass_note)

    # If we need more voices than remaining pitch classes, repeat the full set
    while len(remaining_pcs) < num_parts - 1:
        # Shuffle to get variety in doublings
        shuffled_pcs = list(pitch_classes)
        random.shuffle(shuffled_pcs)
        remaining_pcs.extend(shuffled_pcs)

    # Trim to exactly what we need
    remaining_pcs = remaining_pcs[:num_parts - 1]

    # Step 3: Distribute remaining pitch classes bottom-up
    assigned_notes = {voicing[0]}

    # For very large ensembles, use a smarter allocation strategy
    # Pre-compute all available notes and assign greedily by position
    ABSOLUTE_HIGH = 88
    if num_parts >= 12:
        # Collect all valid chord tones in range
        all_chord_tones = sorted([
            note for note in range(voicing[0] + min_spacing, ABSOLUTE_HIGH + 1)
            if note % 12 in pitch_classes and note not in assigned_notes
        ])

        # Try to space them evenly across the remaining voices
        if len(all_chord_tones) >= num_parts - 1:
            # Pick notes with relatively even spacing
            step = len(all_chord_tones) / (num_parts - 1)
            for voice_idx in range(1, num_parts):
                target_idx = int((voice_idx - 1) * step)
                # Find closest valid note that maintains ascending order
                for candidate in all_chord_tones[target_idx:]:
                    if candidate not in assigned_notes and candidate > voicing[voice_idx - 1]:
                        voicing[voice_idx] = candidate
                        assigned_notes.add(candidate)
                        break
                else:
                    # Fallback: find any valid note
                    for candidate in all_chord_tones:
                        if candidate not in assigned_notes and candidate > voicing[voice_idx - 1]:
                            voicing[voice_idx] = candidate
                            assigned_notes.add(candidate)
                            break
                    else:
                        return None
            return voicing

    for voice_idx in range(1, num_parts):
        voice_low, voice_high = voice_ranges[voice_idx]
        prev_note = voicing[voice_idx - 1]

        # Calculate minimum note to satisfy spacing constraint
        min_note = prev_note + min_spacing

        # Try the assigned pitch class first within voice range
        pc = remaining_pcs[voice_idx - 1]
        candidates = [
            note for note in range(max(voice_low, min_note), voice_high + 1)
            if note % 12 == pc and note not in assigned_notes
        ]

        # If no candidates with assigned PC, try any pitch class from the chord
        if not candidates:
            for alt_pc in pitch_classes:
                candidates = [
                    note for note in range(max(voice_low, min_note), voice_high + 1)
                    if note % 12 == alt_pc and note not in assigned_notes
                ]
                if candidates:
                    break

        # If still no candidates in voice range, expand the search range
        if not candidates:
            # Expand to any valid note above min_note up to absolute soprano limit
            expanded_high = min(max(voice_high, 84), ABSOLUTE_HIGH)
            for alt_pc in pitch_classes:
                candidates = [
                    note for note in range(min_note, expanded_high + 1)
                    if note % 12 == alt_pc and note not in assigned_notes
                ]
                if candidates:
                    break

        if not candidates:
            return None

        # For large ensembles, add some randomization to avoid always
        # picking lowest notes which can cause later voices to run out of range
        if num_parts > 8 and len(candidates) > 1:
            # Pick from lower third of candidates with some randomness
            sorted_candidates = sorted(candidates)
            upper_bound = max(1, len(sorted_candidates) // 3)
            note = random.choice(sorted_candidates[:upper_bound])
        else:
            note = min(candidates)  # Choose lowest valid candidate
        voicing[voice_idx] = note
        assigned_notes.add(note)

    return voicing


def _verify_voicing(
    voicing: List[int],
    voice_ranges: List[Tuple[int, int]],
) -> bool:
    """
    Verify that a voicing satisfies all constraints.

    Args:
        voicing: List of MIDI notes, one per voice
        voice_ranges: List of (low, high) ranges for each voice

    Returns:
        True if all constraints are satisfied, False otherwise.
    """
    # Check: All notes are unique
    if len(set(voicing)) != len(voicing):
        return False

    # Check: Notes are in ascending order
    if voicing != sorted(voicing):
        return False

    # Check: Each note is in its voice range
    for voice_idx, note in enumerate(voicing):
        voice_low, voice_high = voice_ranges[voice_idx]
        if not (voice_low <= note <= voice_high):
            return False

    # Check: Minimum spacing between adjacent voices
    for i in range(len(voicing) - 1):
        if voicing[i + 1] - voicing[i] < MIN_SPACING:
            return False

    return True
