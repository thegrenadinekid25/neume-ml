"""
Non-Chord Tone (NCT) generation module.

This module handles non-chord tones that create time-varying voice events.
Non-chord tones are pitches that are not part of the underlying harmony but
serve as ornaments or voice-leading connections. This module provides utilities
for adding various types of NCTs to voice progressions with configurable density.
"""

from dataclasses import dataclass
from typing import List, Optional
import random


# Non-chord tone types with their characteristics
NON_CHORD_TONE_TYPES = {
    "passing": {
        "description": "Stepwise motion connecting chord tones",
        "approach": "step",
        "resolution": "step",
        "metric_position": "weak"
    },
    "neighbor": {
        "description": "Step away and back to same note",
        "approach": "step",
        "resolution": "step",
        "metric_position": "weak"
    },
    "suspension": {
        "description": "Held note resolves down by step",
        "approach": "hold",
        "resolution": "step_down",
        "metric_position": "strong"
    },
    "anticipation": {
        "description": "Arrives early before chord change",
        "approach": "any",
        "resolution": "same",
        "metric_position": "weak"
    },
    "appoggiatura": {
        "description": "Leap to NCT, resolve by step",
        "approach": "leap",
        "resolution": "step",
        "metric_position": "strong"
    }
}

# Non-chord tone density levels
NON_CHORD_TONE_DENSITY = {
    "none": {
        "probability": 0.0,
        "max_per_voice": 0
    },
    "sparse": {
        "probability": 0.15,
        "max_per_voice": 1
    },
    "moderate": {
        "probability": 0.35,
        "max_per_voice": 2
    },
    "dense": {
        "probability": 0.55,
        "max_per_voice": 3
    }
}


@dataclass
class VoiceEvent:
    """
    Represents a single note event in a voice.

    Attributes:
        pitch: MIDI note number (0-127)
        start_time: Start time in seconds from beginning of chord
        end_time: End time in seconds
    """
    pitch: int
    start_time: float
    end_time: float


def add_non_chord_tones(
    voices: List[int],
    chord_pitches: List[int],
    duration: float,
    density: str,
    seed: Optional[int] = None
) -> List[List[VoiceEvent]]:
    """
    Add non-chord tones to voice events.

    Creates a list of VoiceEvent objects for each voice, starting with the main
    chord tone and optionally inserting non-chord tones based on the specified
    density level. NCT types are chosen randomly, and their pitches are selected
    as scale tones adjacent to the main chord tone.

    Args:
        voices: List of MIDI note numbers, one per voice
        chord_pitches: List of pitch classes (0-11) in the underlying chord
        duration: Total duration of the chord in seconds
        density: Density level for NCTs ("none", "sparse", "moderate", "dense")
        seed: Optional random seed for reproducibility

    Returns:
        List of VoiceEvent lists, one list per voice, with events time-ordered

    Raises:
        ValueError: If density is not a valid density level

    Example:
        >>> voices = [60, 64, 67]  # C, E, G
        >>> chord_pitches = [0, 4, 7]  # C, E, G pitch classes
        >>> duration = 2.0
        >>> events = add_non_chord_tones(voices, chord_pitches, duration, "moderate", seed=42)
        >>> len(events)  # One list per voice
        3
    """
    # Validate density parameter
    if density not in NON_CHORD_TONE_DENSITY:
        raise ValueError(
            f"Invalid density '{density}'. "
            f"Must be one of: {list(NON_CHORD_TONE_DENSITY.keys())}"
        )

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # Get density parameters
    density_config = NON_CHORD_TONE_DENSITY[density]
    nct_probability = density_config["probability"]
    max_per_voice = density_config["max_per_voice"]

    # Initialize result list
    voice_events: List[List[VoiceEvent]] = []

    # Process each voice
    for voice_idx, voice_pitch in enumerate(voices):
        events: List[VoiceEvent] = []

        # Start with the main chord tone event
        main_event = VoiceEvent(
            pitch=voice_pitch,
            start_time=0.0,
            end_time=duration
        )

        # Determine number of NCTs to add to this voice
        nct_count = 0
        if random.random() < nct_probability and max_per_voice > 0:
            nct_count = random.randint(1, max_per_voice)

        if nct_count == 0:
            # No NCTs for this voice
            events.append(main_event)
        else:
            # Generate NCT positions and types
            ncts = _generate_ncts(
                voice_pitch=voice_pitch,
                chord_pitches=chord_pitches,
                duration=duration,
                nct_count=nct_count
            )

            # Convert main event and NCTs into time-ordered events
            events = _split_and_insert_ncts(main_event, ncts)

        voice_events.append(events)

    return voice_events


def _generate_ncts(
    voice_pitch: int,
    chord_pitches: List[int],
    duration: float,
    nct_count: int
) -> List[tuple]:
    """
    Generate non-chord tone specifications for a voice.

    Args:
        voice_pitch: MIDI note of the main voice pitch
        chord_pitches: Pitch classes (0-11) in the chord
        duration: Total duration in seconds
        nct_count: Number of NCTs to generate

    Returns:
        List of tuples: (nct_type, nct_pitch, start_time, duration)
    """
    ncts = []

    # Get available pitch classes adjacent to the main voice
    main_pitch_class = voice_pitch % 12
    adjacent_pitches = _get_adjacent_pitches(main_pitch_class, chord_pitches)

    # Generate NCT timing (distribute across duration)
    time_slots = _generate_time_slots(duration, nct_count)

    for i in range(nct_count):
        # Randomly choose NCT type
        nct_type = random.choice(list(NON_CHORD_TONE_TYPES.keys()))

        # Choose a pitch adjacent to the main chord tone
        if adjacent_pitches:
            nct_pitch_class = random.choice(adjacent_pitches)
            # Find MIDI pitch close to voice pitch
            nct_pitch = _find_adjacent_midi_pitch(voice_pitch, nct_pitch_class)
        else:
            # Fallback: use pitch a semitone away
            nct_pitch = voice_pitch + random.choice([-1, 1])

        # Get timing for this NCT
        start_time, nct_duration = time_slots[i]

        ncts.append((nct_type, nct_pitch, start_time, nct_duration))

    return ncts


def _get_adjacent_pitches(
    main_pitch_class: int,
    chord_pitches: List[int]
) -> List[int]:
    """
    Get pitch classes that are stepwise adjacent to the main pitch.

    Args:
        main_pitch_class: Pitch class (0-11) of the main note
        chord_pitches: Pitch classes in the chord

    Returns:
        List of adjacent pitch classes not in the chord
    """
    adjacent = []

    # Check semitones above and below
    for step in [-1, 1]:
        adjacent_class = (main_pitch_class + step) % 12
        if adjacent_class not in chord_pitches:
            adjacent.append(adjacent_class)

    return adjacent


def _find_adjacent_midi_pitch(main_pitch: int, target_pitch_class: int) -> int:
    """
    Find a MIDI pitch with the target pitch class close to the main pitch.

    Args:
        main_pitch: Main MIDI note number
        target_pitch_class: Target pitch class (0-11)

    Returns:
        MIDI note number with target pitch class
    """
    main_pitch_class = main_pitch % 12

    # Check nearby octaves
    for octave_offset in [0, -12, 12]:
        candidate = (target_pitch_class + octave_offset)
        if main_pitch - 12 <= candidate <= main_pitch + 12:
            return candidate

    # Fallback: use octave closest to main pitch
    if target_pitch_class <= main_pitch_class:
        return target_pitch_class + ((main_pitch // 12) * 12)
    else:
        return target_pitch_class + (((main_pitch // 12) - 1) * 12)


def _generate_time_slots(duration: float, nct_count: int) -> List[tuple]:
    """
    Generate time slot positions for NCTs across the duration.

    Args:
        duration: Total duration in seconds
        nct_count: Number of NCTs to schedule

    Returns:
        List of tuples: (start_time, nct_duration)
    """
    if nct_count == 0:
        return []

    # Generate random positions for NCTs
    positions = sorted([random.uniform(0, duration * 0.8) for _ in range(nct_count)])

    time_slots = []
    nct_duration = min(duration * 0.2, 0.25)  # NCTs typically short relative to chord

    for pos in positions:
        # Ensure NCT doesn't extend past chord duration
        actual_duration = min(nct_duration, duration - pos)
        time_slots.append((pos, actual_duration))

    return time_slots


def _split_and_insert_ncts(
    main_event: VoiceEvent,
    ncts: List[tuple]
) -> List[VoiceEvent]:
    """
    Split the main event and insert NCTs, returning time-ordered events.

    Args:
        main_event: The main VoiceEvent spanning the full duration
        ncts: List of (nct_type, nct_pitch, start_time, duration) tuples

    Returns:
        List of VoiceEvent objects in time order
    """
    if not ncts:
        return [main_event]

    events: List[VoiceEvent] = []
    current_time = main_event.start_time

    # Create a list of all events (main and NCTs) sorted by start time
    all_events = [(main_event.start_time, "main", main_event)]
    for nct_type, nct_pitch, start_time, nct_duration in ncts:
        all_events.append((start_time, "nct", (nct_pitch, start_time, nct_duration)))

    all_events.sort(key=lambda x: x[0])

    # Build events list with proper time segmentation
    for start_idx, (start_time, event_type, event_data) in enumerate(all_events):
        if event_type == "main":
            # Main event - we'll add segments as we encounter NCTs
            pass
        else:
            # NCT event
            nct_pitch, nct_start, nct_duration = event_data

            # Add main event segment before NCT
            if current_time < nct_start:
                events.append(VoiceEvent(
                    pitch=main_event.pitch,
                    start_time=current_time,
                    end_time=nct_start
                ))

            # Add NCT event
            nct_end = nct_start + nct_duration
            events.append(VoiceEvent(
                pitch=nct_pitch,
                start_time=nct_start,
                end_time=nct_end
            ))

            current_time = nct_end

    # Add final segment of main event if needed
    if current_time < main_event.end_time:
        events.append(VoiceEvent(
            pitch=main_event.pitch,
            start_time=current_time,
            end_time=main_event.end_time
        ))

    return events
