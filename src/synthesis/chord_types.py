"""Chord vocabulary and representations for synthesis."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# Note names for human-readable output
ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class ChordQuality(Enum):
    """Chord quality types supported in Phase 1."""

    MAJOR = "maj"
    MINOR = "min"
    DOMINANT_7 = "7"
    # Phase 2 will add: dim, aug, maj7, min7, sus2, sus4, etc.


# Interval patterns for each chord quality (semitones from root)
QUALITY_INTERVALS = {
    ChordQuality.MAJOR: [0, 4, 7],        # Root, major 3rd, perfect 5th
    ChordQuality.MINOR: [0, 3, 7],        # Root, minor 3rd, perfect 5th
    ChordQuality.DOMINANT_7: [0, 4, 7, 10],  # Root, major 3rd, perfect 5th, minor 7th
}


@dataclass
class ChordSpec:
    """Specification for a chord to be synthesized."""

    root: int  # 0-11 (C=0, C#=1, ..., B=11)
    quality: ChordQuality
    bass: Optional[int] = None  # For inversions, None = root position
    extensions: List[str] = field(default_factory=list)  # Future: ["9", "#11", etc.]

    def __post_init__(self):
        """Validate chord specification."""
        if not 0 <= self.root <= 11:
            raise ValueError(f"Root must be 0-11, got {self.root}")
        if self.bass is not None and not 0 <= self.bass <= 11:
            raise ValueError(f"Bass must be 0-11 or None, got {self.bass}")

    def get_pitch_classes(self) -> List[int]:
        """Return list of pitch classes (0-11) in this chord."""
        intervals = QUALITY_INTERVALS[self.quality]
        return [(self.root + interval) % 12 for interval in intervals]

    @property
    def name(self) -> str:
        """Human-readable chord name, e.g., 'Dmaj', 'F#min', 'G7'."""
        root_name = ROOT_NAMES[self.root]
        quality_suffix = self.quality.value

        name = f"{root_name}{quality_suffix}"

        # Add bass note for inversions
        if self.bass is not None and self.bass != self.root:
            bass_name = ROOT_NAMES[self.bass]
            name = f"{name}/{bass_name}"

        return name

    @property
    def root_name(self) -> str:
        """Get the root note name."""
        return ROOT_NAMES[self.root]


def all_chords_phase1() -> List[ChordSpec]:
    """Generate all 36 chord types for Phase 1 (12 roots x 3 qualities)."""
    chords = []
    for root in range(12):
        for quality in [ChordQuality.MAJOR, ChordQuality.MINOR, ChordQuality.DOMINANT_7]:
            chords.append(ChordSpec(root=root, quality=quality))
    return chords
