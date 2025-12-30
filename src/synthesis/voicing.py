"""SATB voice leading logic for chord voicing generation."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .chord_types import ChordQuality, ChordSpec


# Minimum semitones between adjacent voices
MIN_SPACING = 2  # Major 2nd minimum

# Voice ranges as (min, max) MIDI note numbers
BASS_RANGE = (40, 60)     # E2 to C4
TENOR_RANGE = (48, 67)    # C3 to G4
ALTO_RANGE = (55, 74)     # G3 to D5
SOPRANO_RANGE = (60, 79)  # C4 to G5


class VoicingStyle(Enum):
    """Style of voicing for N-voice generation."""

    OPEN = "open"       # Traditional spacing, avoid clusters
    CLOSE = "close"     # Close position, voices within octave
    CLUSTER = "cluster" # Whitacre-style stacked seconds/thirds


@dataclass
class SATBVoicing:
    """SATB voicing with MIDI note numbers for each voice."""

    soprano: int
    alto: int
    tenor: int
    bass: int

    def to_midi_notes(self) -> List[int]:
        """Return notes as list from bass to soprano."""
        return [self.bass, self.tenor, self.alto, self.soprano]

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"S:{self.soprano} A:{self.alto} T:{self.tenor} B:{self.bass}"


@dataclass
class NVoiceVoicing:
    """Voicing for N voices, ordered low to high."""

    notes: List[int]        # MIDI note numbers, ascending order
    voice_count: int
    style: VoicingStyle

    def to_midi_notes(self) -> List[int]:
        """Return notes as list from lowest to highest."""
        return self.notes

    @property
    def lowest(self) -> int:
        """Lowest note in the voicing."""
        return self.notes[0]

    @property
    def highest(self) -> int:
        """Highest note in the voicing."""
        return self.notes[-1]

    @property
    def span(self) -> int:
        """Total span in semitones from bass to soprano."""
        return self.highest - self.lowest

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"{self.voice_count}v {self.style.value}: {'-'.join(str(n) for n in self.notes)}"


def assign_pitch_classes_to_voices(
    pitch_classes: List[int],
    quality: ChordQuality,
) -> Dict[str, int]:
    """
    Decide which pitch class each voice gets.

    For triads (3 pitch classes):
    - Bass: root
    - Tenor: fifth
    - Alto: third
    - Soprano: root (doubled)

    For seventh chords (4 pitch classes):
    - Bass: root
    - Tenor: third
    - Alto: fifth
    - Soprano: seventh

    Args:
        pitch_classes: List of pitch classes in the chord
        quality: Chord quality (for future use in smarter doubling)

    Returns:
        Dict mapping voice name to pitch class
    """
    if len(pitch_classes) == 3:
        # Triad - double the root in soprano
        return {
            "bass": pitch_classes[0],     # root
            "tenor": pitch_classes[2],    # fifth
            "alto": pitch_classes[1],     # third
            "soprano": pitch_classes[0],  # root (doubled)
        }
    elif len(pitch_classes) == 4:
        # Seventh chord - one per voice
        return {
            "bass": pitch_classes[0],     # root
            "tenor": pitch_classes[1],    # third
            "alto": pitch_classes[2],     # fifth
            "soprano": pitch_classes[3],  # seventh
        }
    else:
        raise ValueError(f"Unexpected number of pitch classes: {len(pitch_classes)}")


def _get_all_notes_in_range(pitch_class: int, range_bounds: Tuple[int, int]) -> List[int]:
    """Get all MIDI notes with given pitch class within range."""
    candidates = []
    for midi in range(range_bounds[0], range_bounds[1] + 1):
        if midi % 12 == pitch_class:
            candidates.append(midi)
    return candidates


def _try_build_voicing(
    voice_assignment: Dict[str, int],
    bass_note: int,
    rng: np.random.Generator,
) -> Optional[SATBVoicing]:
    """
    Try to build a voicing from a given bass note.

    Returns None if constraints cannot be satisfied.
    """
    # Step 2: Place tenor (must be > bass + MIN_SPACING)
    tenor_min = bass_note + MIN_SPACING
    tenor_range = (max(tenor_min, TENOR_RANGE[0]), TENOR_RANGE[1])
    if tenor_range[0] > tenor_range[1]:
        return None

    tenor_options = _get_all_notes_in_range(voice_assignment["tenor"], tenor_range)
    if not tenor_options:
        return None
    tenor_note = int(rng.choice(tenor_options))

    # Step 3: Place alto (must be > tenor + MIN_SPACING)
    alto_min = tenor_note + MIN_SPACING
    alto_range = (max(alto_min, ALTO_RANGE[0]), ALTO_RANGE[1])
    if alto_range[0] > alto_range[1]:
        return None

    alto_options = _get_all_notes_in_range(voice_assignment["alto"], alto_range)
    if not alto_options:
        return None
    alto_note = int(rng.choice(alto_options))

    # Step 4: Place soprano (must be > alto + MIN_SPACING)
    soprano_min = alto_note + MIN_SPACING
    soprano_range = (max(soprano_min, SOPRANO_RANGE[0]), SOPRANO_RANGE[1])
    if soprano_range[0] > soprano_range[1]:
        return None

    soprano_options = _get_all_notes_in_range(voice_assignment["soprano"], soprano_range)
    if not soprano_options:
        return None
    soprano_note = int(rng.choice(soprano_options))

    return SATBVoicing(
        soprano=soprano_note,
        alto=alto_note,
        tenor=tenor_note,
        bass=bass_note,
    )


def generate_voicing(chord: ChordSpec, seed: Optional[int] = None) -> SATBVoicing:
    """
    Generate a valid SATB voicing using constraint satisfaction.

    Algorithm:
    1. Get pitch classes from chord
    2. Assign bass note first (root, or specified bass for inversions)
    3. Build upper voices from bottom up, ensuring:
       - Each note is higher than the one below
       - Minimum spacing maintained
       - All chord tones covered
    4. If 3-note chord, double the root in soprano
    5. If constraints fail, try different bass positions

    Hard constraints:
    - No unisons between adjacent voices
    - Bass < Tenor < Alto < Soprano
    - Each voice stays within its range
    - All chord tones represented at least once

    Args:
        chord: The chord specification to voice
        seed: Optional random seed for reproducibility

    Returns:
        SATBVoicing with MIDI note numbers for each voice

    Raises:
        ValueError: If no valid voicing can be found
    """
    rng = np.random.default_rng(seed)
    pitch_classes = chord.get_pitch_classes()

    # Determine bass pitch class (root or specified bass for inversions)
    bass_pc = chord.bass if chord.bass is not None else pitch_classes[0]

    # Assign pitch classes to voices
    voice_assignment = assign_pitch_classes_to_voices(pitch_classes, chord.quality)

    # Override bass pitch class if inversion specified
    voice_assignment["bass"] = bass_pc

    # Get all possible bass notes and try them in random order
    bass_options = _get_all_notes_in_range(voice_assignment["bass"], BASS_RANGE)
    if not bass_options:
        raise ValueError(f"No valid bass notes for pitch class {bass_pc}")

    # Shuffle bass options for variety
    rng.shuffle(bass_options)

    # Try each bass note until we find a working voicing
    for bass_note in bass_options:
        voicing = _try_build_voicing(voice_assignment, bass_note, rng)
        if voicing is not None:
            return voicing

    # If standard assignment fails, try with different upper voice assignments
    # Swap tenor and alto pitch classes
    alt_assignment = voice_assignment.copy()
    alt_assignment["tenor"], alt_assignment["alto"] = (
        alt_assignment["alto"],
        alt_assignment["tenor"],
    )

    rng.shuffle(bass_options)
    for bass_note in bass_options:
        voicing = _try_build_voicing(alt_assignment, bass_note, rng)
        if voicing is not None:
            return voicing

    raise ValueError(
        f"Cannot generate valid voicing for {chord.name} with pitch classes {pitch_classes}"
    )


# =============================================================================
# N-Voice Generation (4-16 parts)
# =============================================================================

# Extended range for N-voice voicings (allows more flexibility)
NVOICE_BASS_FLOOR = 40    # E2
NVOICE_SOPRANO_CEILING = 79  # G5


def _find_note_above(
    pitch_class: int,
    floor: int,
    ceiling: int,
    rng: np.random.Generator,
    style: VoicingStyle,
) -> int:
    """
    Find a note with given pitch class at or above floor.

    For OPEN style, may jump up an octave for variety.
    """
    # Find the first occurrence of pitch_class at or above floor
    note = floor
    while note % 12 != pitch_class:
        note += 1
        if note > ceiling:
            raise ValueError(
                f"No valid note for pitch class {pitch_class} between {floor} and {ceiling}"
            )

    # For OPEN style, maybe jump up an octave for variety
    if style == VoicingStyle.OPEN and rng.random() < 0.3 and note + 12 <= ceiling:
        note += 12

    return note


def _fit_to_range(
    notes: List[int],
    bass_floor: int,
    soprano_ceiling: int,
) -> List[int]:
    """Adjust notes to fit within valid range, maintaining intervals."""
    if not notes:
        return notes

    notes = list(notes)  # Make a copy

    # If highest note exceeds ceiling, shift everything down
    if notes[-1] > soprano_ceiling:
        shift = notes[-1] - soprano_ceiling
        notes = [n - shift for n in notes]

    # If lowest note is below floor, shift up
    if notes[0] < bass_floor:
        shift = bass_floor - notes[0]
        notes = [n + shift for n in notes]

    # Final clamp (shouldn't be needed if logic is correct, but safety)
    notes = [max(bass_floor, min(soprano_ceiling, n)) for n in notes]

    return notes


def _generate_spread_voicing(
    pitch_classes: List[int],
    voice_count: int,
    style: VoicingStyle,
    rng: np.random.Generator,
) -> NVoiceVoicing:
    """
    Generate traditional spread voicing for N voices.

    Strategy:
    1. Determine how many notes per pitch class (distribute voices across chord tones)
    2. Assign octaves from bottom up with strict spacing enforcement
    3. Ensure no duplicate MIDI notes
    """
    n_pitch_classes = len(pitch_classes)

    # Determine spacing based on style
    if style == VoicingStyle.OPEN:
        min_spacing = 3  # Minor 3rd minimum for open
    else:  # CLOSE
        min_spacing = 2  # Major 2nd minimum for close

    # Calculate how many voices per pitch class
    base_count = voice_count // n_pitch_classes
    remainder = voice_count % n_pitch_classes

    # Build list of pitch classes to use, favoring root for doubling
    voice_pcs: List[int] = []
    for i, pc in enumerate(pitch_classes):
        count = base_count
        if i < remainder:
            count += 1
        voice_pcs.extend([pc] * count)

    # Generate notes using a greedy approach from bottom up
    notes: List[int] = []
    current_floor = NVOICE_BASS_FLOOR

    # Sort pitch classes for more natural voice distribution
    # (helps avoid gaps when building from bottom up)
    sorted_pcs = sorted(voice_pcs)

    for pc in sorted_pcs:
        # Find all valid candidates for this pitch class
        candidates = []
        for midi in range(current_floor, NVOICE_SOPRANO_CEILING + 1):
            if midi % 12 == pc:
                candidates.append(midi)

        if candidates:
            # Pick one (prefer lower for denser voicing)
            if style == VoicingStyle.OPEN and len(candidates) > 1 and rng.random() < 0.3:
                note = int(rng.choice(candidates[1:]))  # Skip lowest sometimes
            else:
                note = candidates[0]  # Take lowest valid

            notes.append(note)
            current_floor = note + min_spacing

    # Sort notes (should already be sorted, but ensure)
    notes.sort()

    # Remove any duplicates that slipped through
    unique_notes: List[int] = []
    for note in notes:
        if not unique_notes or note > unique_notes[-1]:
            unique_notes.append(note)
    notes = unique_notes

    # If we don't have enough notes, fill in gaps intelligently
    while len(notes) < voice_count:
        # Find the largest gap and fill it
        best_note = None

        if len(notes) == 0:
            # Start from middle of range
            best_note = 60
        elif len(notes) == 1:
            # Add above or below
            if notes[0] + min_spacing <= NVOICE_SOPRANO_CEILING:
                best_note = notes[0] + min_spacing
            elif notes[0] - min_spacing >= NVOICE_BASS_FLOOR:
                best_note = notes[0] - min_spacing
        else:
            # Find largest gap
            max_gap = 0
            gap_position = -1

            # Check gap below first note
            if notes[0] - NVOICE_BASS_FLOOR > max_gap:
                max_gap = notes[0] - NVOICE_BASS_FLOOR
                gap_position = -1  # Below first note

            # Check gaps between notes
            for i in range(len(notes) - 1):
                gap = notes[i + 1] - notes[i]
                if gap > max_gap:
                    max_gap = gap
                    gap_position = i

            # Check gap above last note
            if NVOICE_SOPRANO_CEILING - notes[-1] > max_gap:
                max_gap = NVOICE_SOPRANO_CEILING - notes[-1]
                gap_position = len(notes)  # Above last note

            # Insert note in the gap
            if gap_position == -1:
                # Below first note
                best_note = notes[0] - min_spacing
                if best_note < NVOICE_BASS_FLOOR:
                    best_note = None
            elif gap_position == len(notes):
                # Above last note
                best_note = notes[-1] + min_spacing
                if best_note > NVOICE_SOPRANO_CEILING:
                    best_note = None
            else:
                # In a gap between notes
                mid = (notes[gap_position] + notes[gap_position + 1]) // 2
                if mid > notes[gap_position] and mid < notes[gap_position + 1]:
                    best_note = mid

        if best_note is not None and best_note not in notes:
            notes.append(best_note)
            notes.sort()
        else:
            # Can't add more notes
            break

    # Final validation - ensure strictly ascending with minimum spacing
    final_notes: List[int] = []
    for note in notes:
        if not final_notes:
            final_notes.append(note)
        elif note >= final_notes[-1] + min_spacing:
            final_notes.append(note)
        # Skip notes that violate spacing

    # Fit to range if needed
    final_notes = _fit_to_range(final_notes, NVOICE_BASS_FLOOR, NVOICE_SOPRANO_CEILING)

    return NVoiceVoicing(notes=final_notes, voice_count=len(final_notes), style=style)


def _generate_cluster_voicing(
    pitch_classes: List[int],
    voice_count: int,
    rng: np.random.Generator,
) -> NVoiceVoicing:
    """
    Generate Whitacre-style cluster voicing.

    Characteristics:
    - Stacked seconds and thirds
    - Voices packed tightly in middle-upper register
    - Creates dense "cloud" effect

    Strategy:
    1. Start from a central pitch (around C4-G4)
    2. Stack voices in seconds/thirds above and below
    3. Weight toward chord tones
    """
    # Cluster center - typically alto/soprano register
    center = int(rng.integers(60, 72))  # C4 to C5

    notes = [center]

    # Intervals to use for building cluster (mix of 2nds, 3rds, 4ths)
    intervals = [2, 3, 4, 2, 3, 5, 2, 4, 3, 2, 3, 4, 2, 3, 4, 2]

    above = center
    below = center

    for i in range(voice_count - 1):
        interval = intervals[i % len(intervals)]

        if i % 2 == 0:
            # Try to add above
            new_above = above + interval
            if new_above <= NVOICE_SOPRANO_CEILING:
                above = new_above
                notes.append(above)
            elif below - interval >= NVOICE_BASS_FLOOR:
                below = below - interval
                notes.append(below)
        else:
            # Try to add below
            new_below = below - interval
            if new_below >= NVOICE_BASS_FLOOR:
                below = new_below
                notes.append(below)
            elif above + interval <= NVOICE_SOPRANO_CEILING:
                above = above + interval
                notes.append(above)

    notes.sort()

    # Ensure minimum spacing (no unisons)
    cleaned_notes: List[int] = []
    for note in notes:
        if not cleaned_notes or note > cleaned_notes[-1]:
            cleaned_notes.append(note)
        elif note + 1 <= NVOICE_SOPRANO_CEILING:
            cleaned_notes.append(note + 1)

    # Pad if needed
    while len(cleaned_notes) < voice_count:
        if cleaned_notes[-1] + 2 <= NVOICE_SOPRANO_CEILING:
            cleaned_notes.append(cleaned_notes[-1] + 2)
        elif cleaned_notes[0] - 2 >= NVOICE_BASS_FLOOR:
            cleaned_notes.insert(0, cleaned_notes[0] - 2)
        else:
            break

    cleaned_notes.sort()

    return NVoiceVoicing(
        notes=cleaned_notes[:voice_count],
        voice_count=voice_count,
        style=VoicingStyle.CLUSTER,
    )


def generate_n_voice_voicing(
    chord: ChordSpec,
    voice_count: int = 4,
    style: VoicingStyle = VoicingStyle.OPEN,
    seed: Optional[int] = None,
) -> NVoiceVoicing:
    """
    Generate a voicing for N voices.

    Args:
        chord: The chord to voice
        voice_count: Number of voices (4, 6, 8, 10, 12, or 16)
        style: OPEN (traditional), CLOSE (compact), or CLUSTER (stacked seconds)
        seed: Random seed for reproducibility

    Returns:
        NVoiceVoicing with notes in ascending order

    Raises:
        ValueError: If voice_count is invalid or voicing cannot be generated
    """
    if voice_count < 4:
        raise ValueError(f"voice_count must be at least 4, got {voice_count}")
    if voice_count > 16:
        raise ValueError(f"voice_count must be at most 16, got {voice_count}")

    rng = np.random.default_rng(seed)
    pitch_classes = chord.get_pitch_classes()

    if style == VoicingStyle.CLUSTER:
        return _generate_cluster_voicing(pitch_classes, voice_count, rng)
    else:
        return _generate_spread_voicing(pitch_classes, voice_count, style, rng)
