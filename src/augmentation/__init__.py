"""
Audio augmentation pipeline for choral synthesis.

This module transforms clean FluidSynth output into realistic choral recordings
by applying acoustic effects calibrated from professional recordings.
"""

from .pitch_scatter import apply_pitch_scatter
from .vibrato import apply_vibrato
from .amplitude_modulation import apply_amplitude_modulation
from .reverb import apply_reverb, ReverbProcessor
from .dynamics import apply_dynamics, measure_lufs, measure_crest_factor
from .lowpass import apply_lowpass, apply_variable_brightness
from .pipeline import (
    AugmentationPipeline,
    AugmentationConfig,
    AUGMENTATION_RANGES,
    augment_audio,
)

__all__ = [
    # Individual effects
    "apply_pitch_scatter",
    "apply_vibrato",
    "apply_amplitude_modulation",
    "apply_reverb",
    "apply_dynamics",
    "apply_lowpass",
    "apply_variable_brightness",
    # Classes
    "ReverbProcessor",
    "AugmentationPipeline",
    "AugmentationConfig",
    # Utilities
    "AUGMENTATION_RANGES",
    "augment_audio",
    "measure_lufs",
    "measure_crest_factor",
]
