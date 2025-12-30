"""
Low-pass filtering for choral synthesis augmentation.

Applies gentle high-frequency rolloff to simulate the warmth of
real choir recordings vs the relative brightness of synthesized audio.
Professional choral recordings typically have significant energy
rolloff above 8-12 kHz depending on recording distance and room.
"""

import numpy as np
from scipy.signal import butter, sosfilt
from typing import Optional


def apply_lowpass(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = 12000.0,
    order: int = 2,
) -> np.ndarray:
    """
    Apply low-pass filter to audio.

    Uses a Butterworth filter for smooth rolloff without resonance.
    Higher order gives steeper rolloff but may introduce phase artifacts.

    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz
        cutoff_hz: Filter cutoff frequency in Hz
        order: Filter order (1-4 recommended, higher = steeper rolloff)

    Returns:
        Filtered audio

    Example:
        >>> audio = np.random.randn(44100) * 0.5
        >>> filtered = apply_lowpass(audio, 44100, cutoff_hz=8000)
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    nyquist = sample_rate / 2

    # Handle edge cases
    if cutoff_hz >= nyquist:
        # Cutoff above Nyquist - no filtering needed
        return audio
    if cutoff_hz <= 0:
        # Invalid cutoff
        return audio

    # Normalize cutoff frequency
    normalized_cutoff = cutoff_hz / nyquist

    # Clamp to valid range (must be < 1.0 for butter)
    normalized_cutoff = min(normalized_cutoff, 0.99)

    # Design Butterworth lowpass filter
    sos = butter(order, normalized_cutoff, btype='low', output='sos')

    # Apply filter
    filtered = sosfilt(sos, audio)

    return filtered


def apply_gentle_rolloff(
    audio: np.ndarray,
    sample_rate: int,
    rolloff_start_hz: float = 8000.0,
    rolloff_amount_db: float = -6.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply gentle high-frequency rolloff simulating recording distance.

    More subtle than a hard lowpass - mimics the natural HF attenuation
    of sound traveling through air and being recorded at a distance.

    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz
        rolloff_start_hz: Frequency where rolloff begins
        rolloff_amount_db: Amount of attenuation at Nyquist (negative dB)
        seed: Random seed (unused, for API consistency)

    Returns:
        Audio with gentle high-frequency rolloff

    Example:
        >>> audio = np.random.randn(44100) * 0.5
        >>> rolled = apply_gentle_rolloff(audio, 44100, 10000, -3)
    """
    # Use a first-order lowpass for gentle rolloff
    # The -3dB point will be at the cutoff frequency
    return apply_lowpass(audio, sample_rate, rolloff_start_hz, order=1)


def apply_variable_brightness(
    audio: np.ndarray,
    sample_rate: int,
    brightness: float = 0.5,
    min_cutoff_hz: float = 5000.0,
    max_cutoff_hz: float = 20000.0,
) -> np.ndarray:
    """
    Apply variable brightness control via low-pass filtering.

    Provides an intuitive 0-1 brightness parameter that maps to
    filter cutoff frequency.

    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz
        brightness: Brightness value 0.0 (warm/dark) to 1.0 (bright)
        min_cutoff_hz: Cutoff at brightness=0
        max_cutoff_hz: Cutoff at brightness=1

    Returns:
        Filtered audio

    Example:
        >>> audio = np.random.randn(44100) * 0.5
        >>> warm = apply_variable_brightness(audio, 44100, brightness=0.2)
        >>> bright = apply_variable_brightness(audio, 44100, brightness=0.9)
    """
    # Clamp brightness to valid range
    brightness = np.clip(brightness, 0.0, 1.0)

    # Exponential mapping for more natural feel
    # (perceived brightness is roughly logarithmic with frequency)
    log_min = np.log10(min_cutoff_hz)
    log_max = np.log10(max_cutoff_hz)
    log_cutoff = log_min + brightness * (log_max - log_min)
    cutoff_hz = 10 ** log_cutoff

    return apply_lowpass(audio, sample_rate, cutoff_hz, order=2)
