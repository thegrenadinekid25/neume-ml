"""
Acoustic analysis toolkit for choral recordings.

Extracts acoustic parameters to inform augmentation pipeline calibration:
- Reverb characteristics (RT60, wet/dry)
- Pitch analysis (vibrato, scatter, drift)
- Spectral profile (formants, HNR, centroid)
- Dynamic range (crest factor, compression)
"""

from .reverb import ReverbAnalysis, analyze_reverb
from .pitch import PitchAnalysis, analyze_pitch
from .spectral import SpectralAnalysis, analyze_spectral
from .dynamics import DynamicsAnalysis, analyze_dynamics
from .report import RecordingAnalysis, generate_report, summarize_analyses

__all__ = [
    "ReverbAnalysis",
    "analyze_reverb",
    "PitchAnalysis",
    "analyze_pitch",
    "SpectralAnalysis",
    "analyze_spectral",
    "DynamicsAnalysis",
    "analyze_dynamics",
    "RecordingAnalysis",
    "generate_report",
    "summarize_analyses",
]
