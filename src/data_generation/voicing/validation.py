"""
Validation module for generated voicing samples.

Ensures generated samples meet all musical and structural constraints.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ValidationError:
    """
    Represents a validation error found in a sample.

    Attributes:
        code: Error code identifying the constraint type (e.g., "UNISON", "SPACING")
        message: Human-readable description of the error
        voice_index: Index of the voice(s) with the issue, if applicable
    """
    code: str
    message: str
    voice_index: Optional[int] = None


def validate_sample(
    voices: List[int],
    metadata: Dict[str, Any]
) -> List[ValidationError]:
    """
    Validate that a generated sample meets all constraints.

    Args:
        voices: List of MIDI note numbers, one per voice
        metadata: Sample metadata containing num_parts and expected_pitch_classes

    Returns:
        Empty list if valid, otherwise list of ValidationError objects
    """
    errors: List[ValidationError] = []

    # VOICE_COUNT: Check number of voices matches metadata
    expected_parts = metadata.get("num_parts", len(voices))
    if len(voices) != expected_parts:
        errors.append(ValidationError(
            code="VOICE_COUNT",
            message=f"Expected {expected_parts} voices, got {len(voices)}"
        ))

    # UNISON: Check for duplicate MIDI notes
    seen_notes = set()
    for idx, note in enumerate(voices):
        if note in seen_notes:
            errors.append(ValidationError(
                code="UNISON",
                message=f"Duplicate note {note} (MIDI) found in voices",
                voice_index=idx
            ))
            break  # Report once
        seen_notes.add(note)

    # ASCENDING: Check voices are in ascending order
    for idx in range(len(voices) - 1):
        if voices[idx] >= voices[idx + 1]:
            errors.append(ValidationError(
                code="ASCENDING",
                message=f"Voices not in ascending order at position {idx}",
                voice_index=idx
            ))
            break  # Report once

    # SPACING: Check minimum spacing between adjacent voices
    # For larger ensembles (>10 voices), allow semitone spacing
    min_spacing = 1 if len(voices) > 10 else 2
    for idx in range(len(voices) - 1):
        spacing = voices[idx + 1] - voices[idx]
        if spacing < min_spacing:
            errors.append(ValidationError(
                code="SPACING",
                message=f"Insufficient spacing ({spacing} semitones) between voices {idx} and {idx + 1}",
                voice_index=idx
            ))
            break  # Report once

    # RANGE: Check notes within general choral range
    # For N-voice textures (4-16 voices), we use a general range
    # that covers bass to soprano: MIDI 36 (C2) to 88 (E6)
    # Extended range allows for larger ensembles
    GENERAL_LOW = 36   # C2 - low bass
    GENERAL_HIGH = 88  # E6 - extended soprano for large ensembles

    for idx, note in enumerate(voices):
        if note < GENERAL_LOW or note > GENERAL_HIGH:
            errors.append(ValidationError(
                code="RANGE",
                message=f"Voice {idx} note {note} outside general choral range {GENERAL_LOW}-{GENERAL_HIGH}",
                voice_index=idx
            ))

    # PITCH_CLASSES: Check all expected pitch classes are present
    expected_pitch_classes = metadata.get("expected_pitch_classes", [])
    if expected_pitch_classes:
        actual_pitch_classes = set(note % 12 for note in voices)
        expected_set = set(expected_pitch_classes)
        missing = expected_set - actual_pitch_classes
        if missing:
            errors.append(ValidationError(
                code="PITCH_CLASSES",
                message=f"Missing expected pitch classes: {sorted(missing)}"
            ))

    return errors


def is_valid_sample(
    voices: List[int],
    metadata: Dict[str, Any]
) -> bool:
    """
    Convenience function to check if a sample is valid.

    Args:
        voices: List of MIDI note numbers, one per voice
        metadata: Sample metadata

    Returns:
        True if sample is valid (no errors), False otherwise
    """
    return len(validate_sample(voices, metadata)) == 0
