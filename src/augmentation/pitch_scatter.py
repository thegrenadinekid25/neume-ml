"""
Pitch scatter augmentation for choral synthesis.

Simulates multiple singers per part who don't match pitch exactly.
Real choral recordings show ±16-50 cents of pitch scatter per voice.
"""

import numpy as np
from scipy.signal import resample
from scipy.ndimage import gaussian_filter1d
from typing import Optional


def apply_pitch_scatter(
    audio: np.ndarray,
    sample_rate: int,
    std_cents: float,
    drift_rate_hz: float = 0.3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply random pitch deviation to simulate imperfect unison.

    When multiple singers perform the same part, their pitches naturally
    deviate slightly from each other. This creates the characteristic
    "warmth" of choral sound. The deviation is slow-moving (not jittery)
    which simulates natural pitch drift in sustained singing.

    Args:
        audio: Single voice audio [samples]
        sample_rate: Audio sample rate
        std_cents: Standard deviation of pitch scatter in cents
        drift_rate_hz: How fast the pitch wanders (slow = realistic)
        seed: Random seed for reproducibility

    Returns:
        Pitch-scattered audio

    Example:
        >>> audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> scattered = apply_pitch_scatter(audio, 44100, std_cents=30)
    """
    if std_cents <= 0:
        return audio

    rng = np.random.default_rng(seed)
    duration = len(audio) / sample_rate
    num_control_points = max(2, int(duration * drift_rate_hz * 2))

    # Generate smooth random pitch curve using control points
    control_points = rng.normal(0, std_cents, num_control_points)
    t_control = np.linspace(0, duration, num_control_points)
    t_samples = np.linspace(0, duration, len(audio))

    # Interpolate to sample rate
    pitch_curve_cents = np.interp(t_samples, t_control, control_points)

    # Apply smoothing to avoid artifacts
    sigma = sample_rate * 0.05  # 50ms smoothing
    pitch_curve_cents = gaussian_filter1d(pitch_curve_cents, sigma=sigma)

    # Convert cents to ratio: ratio = 2^(cents/1200)
    ratio_curve = np.power(2, pitch_curve_cents / 1200)

    # Apply time-varying pitch shift via block-based resampling
    return _block_pitch_shift(audio, ratio_curve, sample_rate, block_size=2048)


def _block_pitch_shift(
    audio: np.ndarray,
    ratio_curve: np.ndarray,
    sample_rate: int,
    block_size: int = 2048,
) -> np.ndarray:
    """
    Block-based pitch shifting with overlap-add.

    Uses resampling to shift pitch while maintaining duration.
    Process in overlapping blocks to handle time-varying pitch.

    Args:
        audio: Input audio signal
        ratio_curve: Time-varying pitch ratio (1.0 = no change)
        sample_rate: Sample rate
        block_size: Size of processing blocks

    Returns:
        Pitch-shifted audio
    """
    hop_size = block_size // 4
    num_blocks = (len(audio) - block_size) // hop_size + 1

    output = np.zeros(len(audio), dtype=audio.dtype)
    window = np.hanning(block_size)
    normalization = np.zeros(len(audio))

    for i in range(num_blocks):
        start = i * hop_size
        end = start + block_size

        if end > len(audio):
            break

        block = audio[start:end] * window
        ratio = ratio_curve[start + block_size // 2]

        # Skip if ratio is essentially 1 (no change needed)
        if abs(ratio - 1.0) < 0.0001:
            output[start:end] += block
            normalization[start:end] += window
            continue

        # Resample block to shift pitch
        # To raise pitch by ratio, resample to shorter length then stretch back
        new_length = int(block_size / ratio)
        if new_length > 0 and new_length < block_size * 3:  # Sanity check
            # Resample to new length (changes pitch)
            shifted = resample(block, new_length)
            # Resample back to original length (maintains duration)
            shifted = resample(shifted, block_size)
            output[start:end] += shifted
            normalization[start:end] += window

    # Normalize by window overlap
    normalization = np.maximum(normalization, 1e-8)
    output = output / normalization

    return output


def apply_pitch_scatter_multivoice(
    audio: np.ndarray,
    sample_rate: int,
    std_cents: float,
    num_layers: int = 3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply pitch scatter by layering multiple independently detuned copies.

    This creates a more realistic ensemble effect than single-voice scatter.
    Each layer gets independent random detuning.

    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        std_cents: Standard deviation of scatter in cents
        num_layers: Number of detuned copies to mix
        seed: Random seed for reproducibility

    Returns:
        Layered audio with ensemble effect
    """
    if num_layers <= 1:
        return apply_pitch_scatter(audio, sample_rate, std_cents, seed=seed)

    rng = np.random.default_rng(seed)
    layers = []

    for i in range(num_layers):
        layer_seed = rng.integers(0, 2**31) if seed is not None else None
        # Reduce scatter per layer since they accumulate
        layer_scatter = std_cents / np.sqrt(num_layers)
        layer = apply_pitch_scatter(
            audio, sample_rate, layer_scatter, seed=layer_seed
        )
        layers.append(layer)

    # Mix layers with equal weight
    output = np.mean(layers, axis=0)

    return output
