"""
Distribution weights for sampling parameters in voicing generation.

Defines probability distributions for various voicing attributes to ensure
diverse and musically representative sample generation.
"""

import random
from typing import Dict, Optional


# Distribution of distinct pitch ratios relative to chord size
DISTRIBUTION_WEIGHTS: Dict[str, float] = {
    "full": 0.40,        # Use all chord tones
    "minus_one": 0.25,   # Omit one tone
    "minus_two": 0.20,   # Omit two tones
    "shell": 0.15,       # Minimal shell voicing (2-3 tones)
}

# Chord type frequency distribution (weighted toward common triads/seventh chords)
CHORD_FREQUENCY: Dict[str, float] = {
    # Common triads
    "major": 0.15,
    "minor": 0.12,
    "diminished": 0.02,
    "augmented": 0.01,

    # Common seventh chords
    "dom7": 0.12,
    "maj7": 0.08,
    "min7": 0.10,
    "half_dim7": 0.03,
    "dim7": 0.02,
    "aug7": 0.01,

    # Extended chords
    "maj9": 0.04,
    "min9": 0.03,
    "dom9": 0.05,
    "dom11": 0.02,
    "dom13": 0.02,
    "maj13": 0.02,

    # Upper structure/modal chords
    "sus2": 0.03,
    "sus4": 0.04,
    "add9": 0.02,
    "add11": 0.01,

    # Jazz extensions
    "maj7#11": 0.02,
    "dom7#11": 0.02,
    "dom7b9": 0.02,
    "dom7sharp9": 0.02,

    # Sparse/minimal
    "power_chord": 0.02,
    "dyad": 0.01,

    # Experimental/rare
    "cluster_chromatic": 0.01,
    "whole_tone": 0.01,
}

# Bass note inversion distribution
BASS_NOTE_WEIGHTS: Dict[str, float] = {
    "root": 0.50,           # Root position
    "first_inv": 0.20,      # First inversion
    "second_inv": 0.15,     # Second inversion
    "third_inv": 0.10,      # Third inversion (for seventh chords)
    "slash": 0.05,          # Slash chord (non-chord tone bass)
}

# Voicing style distribution
VOICING_STYLE_WEIGHTS: Dict[str, float] = {
    "open": 0.40,           # Wide spacing, multiple octaves
    "closed": 0.25,         # Compact, within octave
    "mixed": 0.25,          # Mix of open and closed intervals
    "wide": 0.10,           # Extremely spread out
}

# Duration/note length distribution
DURATION_WEIGHTS: Dict[str, float] = {
    "very_short": 0.10,     # Eighth note or faster
    "short": 0.20,          # Quarter note
    "medium": 0.40,         # Half note
    "long": 0.20,           # Whole note
    "very_long": 0.10,      # Multiple measures
}

# Non-chord tone (NCT) density distribution
NCT_DENSITY_WEIGHTS: Dict[str, float] = {
    "none": 0.30,           # No passing/neighboring tones
    "sparse": 0.35,         # Occasional NCTs (1-2 per measure)
    "moderate": 0.25,       # Regular NCTs (3-4 per measure)
    "dense": 0.10,          # Heavy use of NCTs (5+ per measure)
}


def weighted_choice(
    options: Dict[str, float],
    seed: Optional[int] = None
) -> str:
    """
    Select a random option weighted by its probability value.

    Values do not need to sum to 1.0; they are normalized internally.

    Args:
        options: Dictionary mapping choice names to weight values
        seed: Optional random seed for reproducibility

    Returns:
        Selected option name

    Raises:
        ValueError: If options is empty or all weights are zero
    """
    if not options:
        raise ValueError("options dictionary cannot be empty")

    if seed is not None:
        random.seed(seed)

    # Normalize weights
    total = sum(options.values())
    if total <= 0:
        raise ValueError("all weights must sum to a positive value")

    normalized = {k: v / total for k, v in options.items()}

    # Use random.choices for weighted selection
    choice = random.choices(
        list(normalized.keys()),
        weights=list(normalized.values()),
        k=1
    )
    return choice[0]
