"""
Dynamics analysis for choral recordings.

Analyzes dynamic range, compression characteristics, and loudness.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DynamicsAnalysis:
    """Dynamic characteristics extracted from audio."""
    peak_db: float                    # Maximum level (dBFS)
    rms_db: float                     # Average RMS level (dBFS)
    crest_factor_db: float            # Peak - RMS (lower = more compressed)
    dynamic_range_db: float           # Difference between loud and soft passages
    lufs_integrated: float            # Integrated loudness (LUFS)
    compression_estimate: str         # "heavy", "moderate", "light", "none"
    loudness_range_lu: float          # Loudness range (LU)


def analyze_dynamics(audio: np.ndarray, sr: int) -> DynamicsAnalysis:
    """
    Analyze dynamic characteristics.

    Crest factor interpretation:
    - < 10 dB: heavily compressed (broadcast, some pop)
    - 10-14 dB: moderate compression (typical mastered classical)
    - 14-20 dB: light compression (audiophile classical)
    - > 20 dB: essentially uncompressed

    For choral recordings, expect 12-18 dB typically.

    Args:
        audio: Mono audio signal (normalized to -1 to 1)
        sr: Sample rate in Hz

    Returns:
        DynamicsAnalysis with extracted parameters
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Basic measurements
    peak_linear = np.max(np.abs(audio))
    peak_db = 20 * np.log10(peak_linear + 1e-10)

    rms_linear = np.sqrt(np.mean(audio ** 2))
    rms_db = 20 * np.log10(rms_linear + 1e-10)

    crest_factor_db = peak_db - rms_db

    # Dynamic range (difference between loud and soft passages)
    dynamic_range_db = _calculate_dynamic_range(audio, sr)

    # LUFS measurement
    lufs, loudness_range = _calculate_lufs(audio, sr)

    # Classify compression level
    compression_estimate = _classify_compression(crest_factor_db, dynamic_range_db)

    return DynamicsAnalysis(
        peak_db=float(peak_db),
        rms_db=float(rms_db),
        crest_factor_db=float(crest_factor_db),
        dynamic_range_db=float(dynamic_range_db),
        lufs_integrated=float(lufs),
        compression_estimate=compression_estimate,
        loudness_range_lu=float(loudness_range)
    )


def _calculate_dynamic_range(audio: np.ndarray, sr: int) -> float:
    """
    Calculate dynamic range as difference between loud and soft passages.

    Uses short-term RMS to identify passage levels.
    """
    # Calculate short-term RMS (400ms windows, typical for loudness)
    window_samples = int(sr * 0.4)
    hop_samples = int(sr * 0.1)

    if len(audio) < window_samples:
        return 0.0

    # Calculate RMS for each window
    n_frames = (len(audio) - window_samples) // hop_samples + 1
    rms_values = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_samples
        end = start + window_samples
        frame = audio[start:end]
        rms_values[i] = np.sqrt(np.mean(frame ** 2))

    # Convert to dB
    rms_db = 20 * np.log10(rms_values + 1e-10)

    # Filter out very quiet passages (likely silence)
    threshold = np.max(rms_db) - 60  # 60dB below peak
    active_mask = rms_db > threshold

    if np.sum(active_mask) < 3:
        return 0.0

    active_rms = rms_db[active_mask]

    # Dynamic range: 95th percentile - 5th percentile
    # This avoids outliers from transients and near-silence
    dr = np.percentile(active_rms, 95) - np.percentile(active_rms, 5)

    return dr


def _calculate_lufs(audio: np.ndarray, sr: int) -> Tuple[float, float]:
    """
    Calculate integrated LUFS and loudness range.

    Implements ITU-R BS.1770 loudness measurement (simplified).
    """
    # K-weighting filter (simplified version)
    # Stage 1: High shelf filter
    # Stage 2: High-pass filter (RLB weighting)

    audio_weighted = _apply_k_weighting(audio, sr)

    # Calculate mean square
    mean_square = np.mean(audio_weighted ** 2)

    # Convert to LUFS
    # LUFS = -0.691 + 10 * log10(mean_square)
    if mean_square > 0:
        lufs = -0.691 + 10 * np.log10(mean_square)
    else:
        lufs = -70.0  # Very quiet

    # Loudness range (simplified)
    loudness_range = _calculate_loudness_range(audio_weighted, sr)

    return lufs, loudness_range


def _apply_k_weighting(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply K-weighting filter for loudness measurement.

    This is a simplified implementation of ITU-R BS.1770 K-weighting.
    """
    from scipy.signal import butter, sosfilt

    # High shelf filter at 1500 Hz (+4 dB)
    # Approximated with a simple high-pass boost
    # For simplicity, we use a high-shelf approximation

    # Stage 1: Pre-filter (high shelf)
    # Boost high frequencies by ~4 dB above 1500 Hz
    nyquist = sr / 2

    if nyquist > 1500:
        # High-pass at 150 Hz
        sos_hp = butter(2, 150 / nyquist, btype='high', output='sos')
        audio = sosfilt(sos_hp, audio)

        # Simple high-shelf approximation using parallel high-pass
        sos_shelf = butter(2, 1500 / nyquist, btype='high', output='sos')
        high_boost = sosfilt(sos_shelf, audio) * 0.58  # ~4dB boost factor
        audio = audio + high_boost

    # Stage 2: RLB (Revised Low-frequency B-weighting)
    # High-pass at ~60 Hz
    if nyquist > 60:
        sos_rlb = butter(2, 60 / nyquist, btype='high', output='sos')
        audio = sosfilt(sos_rlb, audio)

    return audio


def _calculate_loudness_range(audio_weighted: np.ndarray, sr: int) -> float:
    """
    Calculate loudness range (LRA) in LU.

    Based on EBU R128 loudness range calculation.
    """
    # Short-term loudness (3s windows, 1s overlap)
    window_samples = int(sr * 3.0)
    hop_samples = int(sr * 1.0)

    if len(audio_weighted) < window_samples:
        return 0.0

    n_frames = (len(audio_weighted) - window_samples) // hop_samples + 1
    short_term_loudness = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_samples
        end = start + window_samples
        frame = audio_weighted[start:end]
        ms = np.mean(frame ** 2)
        if ms > 0:
            short_term_loudness[i] = -0.691 + 10 * np.log10(ms)
        else:
            short_term_loudness[i] = -70.0

    # Gate at -70 LUFS (absolute) and relative gate
    absolute_gate = -70.0
    active_mask = short_term_loudness > absolute_gate

    if np.sum(active_mask) < 3:
        return 0.0

    active_loudness = short_term_loudness[active_mask]

    # Relative gate: -20 LU below ungated average
    ungated_avg = np.mean(active_loudness)
    relative_gate = ungated_avg - 20

    final_mask = active_loudness > relative_gate
    if np.sum(final_mask) < 3:
        return 0.0

    gated_loudness = active_loudness[final_mask]

    # LRA: 95th percentile - 10th percentile
    lra = np.percentile(gated_loudness, 95) - np.percentile(gated_loudness, 10)

    return lra


def _classify_compression(crest_factor: float, dynamic_range: float) -> str:
    """
    Classify compression level based on crest factor and dynamic range.

    Crest factor interpretation for choral:
    - < 10 dB: heavily compressed
    - 10-14 dB: moderate compression
    - 14-20 dB: light compression
    - > 20 dB: essentially uncompressed

    Dynamic range is used as secondary indicator.
    """
    # Primary classification from crest factor
    if crest_factor < 10:
        primary = "heavy"
    elif crest_factor < 14:
        primary = "moderate"
    elif crest_factor < 20:
        primary = "light"
    else:
        primary = "none"

    # Adjust based on dynamic range
    # If DR is very low despite moderate crest factor, increase compression estimate
    if dynamic_range < 6 and primary in ("light", "none"):
        return "moderate"
    elif dynamic_range < 3:
        return "heavy"

    return primary
