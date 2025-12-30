"""
Dynamics normalization for choral synthesis augmentation.

Normalizes loudness and adjusts dynamic range to match real recordings.
Real choral recordings show LUFS of -33 to -20 with crest factors of 15-24 dB.
"""

import numpy as np
from typing import Tuple


def measure_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """
    Measure integrated LUFS (simplified ITU-R BS.1770).

    LUFS (Loudness Units Full Scale) is the standard for measuring
    perceived loudness. This is a simplified implementation that
    provides reasonable estimates.

    Args:
        audio: Audio signal (mono or stereo)
        sample_rate: Sample rate in Hz

    Returns:
        LUFS value (typically -70 to 0)

    Example:
        >>> audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)) * 0.5
        >>> lufs = measure_lufs(audio, 44100)
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Apply K-weighting filter (simplified)
    audio_weighted = _apply_k_weighting(audio, sample_rate)

    # Calculate mean square
    mean_square = np.mean(audio_weighted ** 2)

    if mean_square <= 0:
        return -70.0

    # Convert to LUFS
    # LUFS = -0.691 + 10 * log10(mean_square)
    lufs = -0.691 + 10 * np.log10(mean_square)

    return float(lufs)


def _apply_k_weighting(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Apply K-weighting filter for loudness measurement.

    K-weighting approximates human perception of loudness by boosting
    high frequencies and cutting low frequencies.
    """
    from scipy.signal import butter, sosfilt

    nyquist = sample_rate / 2

    # High-pass at 60 Hz (RLB weighting)
    if nyquist > 60:
        sos_hp = butter(2, 60 / nyquist, btype="high", output="sos")
        audio = sosfilt(sos_hp, audio)

    # High shelf boost at 1500 Hz (~4 dB)
    if nyquist > 1500:
        sos_shelf = butter(2, 1500 / nyquist, btype="high", output="sos")
        high_boost = sosfilt(sos_shelf, audio) * 0.58  # ~4dB boost factor
        audio = audio + high_boost

    return audio


def measure_crest_factor(audio: np.ndarray) -> float:
    """
    Measure crest factor (peak-to-RMS ratio in dB).

    Crest factor indicates dynamic range. Higher values mean more
    dynamic (uncompressed) audio.

    Typical values:
    - < 10 dB: heavily compressed
    - 10-14 dB: moderate compression
    - 14-20 dB: light compression
    - > 20 dB: essentially uncompressed

    Args:
        audio: Audio signal

    Returns:
        Crest factor in dB

    Example:
        >>> audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> cf = measure_crest_factor(audio)  # Should be ~3 dB for sine
    """
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))

    if rms <= 0:
        return 0.0

    return float(20 * np.log10(peak / rms))


def apply_dynamics(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float,
    target_crest_factor_db: float,
) -> np.ndarray:
    """
    Normalize loudness and adjust dynamic range.

    First adjusts overall loudness to target LUFS, then adjusts
    dynamic range (crest factor) through soft compression or expansion.

    Args:
        audio: Input audio
        sample_rate: Audio sample rate
        target_lufs: Target loudness in LUFS (-33 to -20 for choral)
        target_crest_factor_db: Target crest factor (15-24 dB for choral)

    Returns:
        Dynamics-processed audio

    Example:
        >>> audio = np.random.randn(44100) * 0.1
        >>> normalized = apply_dynamics(audio, 44100, target_lufs=-26, target_crest_factor_db=18)
    """
    # Ensure mono for processing
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Measure current levels
    current_lufs = measure_lufs(audio, sample_rate)
    current_crest = measure_crest_factor(audio)

    # Apply gain to match target LUFS
    gain_db = target_lufs - current_lufs
    # Limit gain to reasonable range to avoid extreme amplification
    gain_db = np.clip(gain_db, -40, 40)
    audio = audio * np.power(10, gain_db / 20)

    # Adjust crest factor via soft compression/expansion
    crest_diff = target_crest_factor_db - current_crest

    if abs(crest_diff) > 1:  # Only adjust if significant difference
        audio = _adjust_crest_factor(audio, crest_diff)

    # Final peak normalization to prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0.99:
        audio = audio * (0.99 / peak)

    return audio


def _adjust_crest_factor(audio: np.ndarray, crest_diff_db: float) -> np.ndarray:
    """
    Adjust crest factor via soft compression or expansion.

    Positive crest_diff means we need to increase dynamics (expand).
    Negative crest_diff means we need to decrease dynamics (compress).

    Args:
        audio: Input audio
        crest_diff_db: How much to change crest factor (+ = more dynamic)

    Returns:
        Processed audio with adjusted crest factor
    """
    if abs(crest_diff_db) < 0.5:
        return audio

    if crest_diff_db > 0:
        # Expand dynamics (increase crest factor)
        # Apply gentle expansion to peaks
        threshold_percentile = 90
        threshold = np.percentile(np.abs(audio), threshold_percentile)
        ratio = 1.0 + crest_diff_db / 30  # Gentle expansion

        expanded = audio.copy()
        mask = np.abs(audio) > threshold
        expanded[mask] = np.sign(audio[mask]) * (
            threshold + (np.abs(audio[mask]) - threshold) * ratio
        )

        # Renormalize to original peak
        expanded_max = np.max(np.abs(expanded))
        if expanded_max > 0:
            peak_ratio = np.max(np.abs(audio)) / expanded_max
            return expanded * peak_ratio
        return expanded
    else:
        # Compress dynamics (decrease crest factor)
        threshold_percentile = 80
        threshold = np.percentile(np.abs(audio), threshold_percentile)
        ratio = 1.0 / (1.0 - crest_diff_db / 30)  # Gentle compression

        compressed = audio.copy()
        mask = np.abs(audio) > threshold
        compressed[mask] = np.sign(audio[mask]) * (
            threshold + (np.abs(audio[mask]) - threshold) / ratio
        )
        return compressed


def normalize_peak(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """
    Normalize audio to target peak level.

    Args:
        audio: Input audio
        target_db: Target peak level in dBFS

    Returns:
        Peak-normalized audio
    """
    current_peak = np.max(np.abs(audio))
    if current_peak <= 0:
        return audio

    target_linear = np.power(10, target_db / 20)
    return audio * (target_linear / current_peak)


def normalize_rms(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target RMS level.

    Args:
        audio: Input audio
        target_db: Target RMS level in dBFS

    Returns:
        RMS-normalized audio
    """
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms <= 0:
        return audio

    target_linear = np.power(10, target_db / 20)
    normalized = audio * (target_linear / current_rms)

    # Prevent clipping
    peak = np.max(np.abs(normalized))
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)

    return normalized


def apply_soft_compression(
    audio: np.ndarray,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Apply soft-knee compression to audio.

    More sophisticated compression with attack and release times.
    Useful for controlling peaks while maintaining natural dynamics.

    Args:
        audio: Input audio
        threshold_db: Compression threshold in dBFS
        ratio: Compression ratio (4:1 means 4dB input becomes 1dB above threshold)
        attack_ms: Attack time in milliseconds
        release_ms: Release time in milliseconds
        sample_rate: Sample rate

    Returns:
        Compressed audio
    """
    from scipy.ndimage import uniform_filter1d

    # Convert to amplitude envelope
    envelope = np.abs(audio)

    # Smooth envelope for gain calculation
    attack_samples = int(attack_ms * sample_rate / 1000)
    release_samples = int(release_ms * sample_rate / 1000)

    # Use running max for attack, running average for release
    # Simplified approach: use uniform filter
    smoothed_env = uniform_filter1d(envelope, size=attack_samples)
    smoothed_env = np.maximum(smoothed_env, uniform_filter1d(envelope, size=release_samples))

    # Convert to dB
    env_db = 20 * np.log10(smoothed_env + 1e-10)

    # Calculate gain reduction
    gain_reduction_db = np.zeros_like(env_db)
    above_threshold = env_db > threshold_db
    gain_reduction_db[above_threshold] = (
        (env_db[above_threshold] - threshold_db) * (1 - 1 / ratio)
    )

    # Apply gain reduction
    gain_linear = np.power(10, -gain_reduction_db / 20)
    compressed = audio * gain_linear

    return compressed


def apply_loudness_variation(
    audio: np.ndarray,
    sample_rate: int,
    variation_db: float = 3.0,
    rate_hz: float = 0.1,
    seed: int = None,
) -> np.ndarray:
    """
    Apply slow loudness variation to simulate natural performance dynamics.

    Real recordings have slow volume variations from performance dynamics.
    This adds subtle, slow modulation to simulate this effect.

    Args:
        audio: Input audio
        sample_rate: Sample rate
        variation_db: Amount of level variation in dB
        rate_hz: Rate of variation (very slow, ~0.1 Hz)
        seed: Random seed for reproducibility

    Returns:
        Audio with natural loudness variation
    """
    rng = np.random.default_rng(seed)

    duration = len(audio) / sample_rate
    t = np.linspace(0, duration, len(audio))

    # Generate slow, smooth loudness envelope
    phase = rng.uniform(0, 2 * np.pi)
    modulation_db = variation_db * np.sin(2 * np.pi * rate_hz * t + phase)

    # Add some randomness for natural feel
    num_points = max(5, int(duration * 0.5))
    random_points = rng.uniform(-variation_db * 0.3, variation_db * 0.3, num_points)
    t_points = np.linspace(0, duration, num_points)
    random_component = np.interp(t, t_points, random_points)

    total_modulation_db = modulation_db + random_component

    # Convert to linear gain
    gain = np.power(10, total_modulation_db / 20)

    return audio * gain
