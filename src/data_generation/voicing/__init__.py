"""Voicing generation modules."""

from .chord_types import CHORD_TYPES, ROOT_NAMES, get_pitch_classes, categorize_intervals
from .bass_note import BASS_NOTE_TYPES, select_bass_note
from .shell_strategies import SHELL_STRATEGIES, select_pitches
from .duration import DURATION_OPTIONS, select_duration
from .voicing_styles import VOICING_STYLES, apply_voicing_style
from .non_chord_tones import NON_CHORD_TONE_TYPES, NON_CHORD_TONE_DENSITY, add_non_chord_tones
from .voice_distribution import distribute_voices, compute_voice_ranges, REGISTERS
from .validation import validate_sample, is_valid_sample
from .weights import (
    DISTRIBUTION_WEIGHTS,
    CHORD_FREQUENCY,
    BASS_NOTE_WEIGHTS,
    VOICING_STYLE_WEIGHTS,
    DURATION_WEIGHTS,
    NCT_DENSITY_WEIGHTS,
    weighted_choice,
)

__all__ = [
    # Chord types
    "CHORD_TYPES",
    "ROOT_NAMES",
    "get_pitch_classes",
    "categorize_intervals",
    # Bass note
    "BASS_NOTE_TYPES",
    "select_bass_note",
    # Shell strategies
    "SHELL_STRATEGIES",
    "select_pitches",
    # Duration
    "DURATION_OPTIONS",
    "select_duration",
    # Voicing styles
    "VOICING_STYLES",
    "apply_voicing_style",
    # Non-chord tones
    "NON_CHORD_TONE_TYPES",
    "NON_CHORD_TONE_DENSITY",
    "add_non_chord_tones",
    # Voice distribution
    "distribute_voices",
    "compute_voice_ranges",
    "REGISTERS",
    # Validation
    "validate_sample",
    "is_valid_sample",
    # Weights
    "DISTRIBUTION_WEIGHTS",
    "CHORD_FREQUENCY",
    "BASS_NOTE_WEIGHTS",
    "VOICING_STYLE_WEIGHTS",
    "DURATION_WEIGHTS",
    "NCT_DENSITY_WEIGHTS",
    "weighted_choice",
]
