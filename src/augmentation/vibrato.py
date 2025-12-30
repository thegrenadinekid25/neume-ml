"""
Vibrato augmentation for choral synthesis.

Applies per-voice pitch modulation simulating natural voice production.
Real choral recordings show vibrato rates of 4.6-7 Hz with 20-65 cents depth.
"""

import numpy as np
from typing import Optional

from .pitch_scatter import _block_pitch_shift


def apply_vibrato(
    audio: np.ndarray,
    sample_rate: int,
    rate_hz: float,
    depth_cents: float,
    onset_delay_sec: float = 0.15,
    onset_ramp_sec: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply sinusoidal vibrato with delayed onset.

    Real singers don't start vibrato immediately - it develops after the
    note onset. This function models that natural behavior with a delayed
    onset and gradual ramp-up.

    The vibrato uses a slightly randomized rate to avoid sounding mechanical.

    Args:
        audio: Single voice audio [samples]
        sample_rate: Audio sample rate
        rate_hz: Vibrato frequency (typically 4.6-7 Hz for choral)
        depth_cents: Vibrato amplitude in cents (typically 20-65 cents)
        onset_delay_sec: Time before vibrato starts (typically 0.1-0.3s)
        onset_ramp_sec: Time to ramp up to full depth
        seed: Random seed for reproducibility

    Returns:
        Audio with vibrato applied

    Example:
        >>> audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 88200))
        >>> vibrato = apply_vibrato(audio, 44100, rate_hz=5.5, depth_cents=40)
    """
    if depth_cents <= 0 or rate_hz <= 0:
        return audio

    rng = np.random.default_rng(seed)

    duration = len(audio) / sample_rate
    t = np.linspace(0, duration, len(audio))

    # Add slight randomness to rate for natural feel
    rate_variation = rng.uniform(0.95, 1.05)
    actual_rate = rate_hz * rate_variation

    # Add random starting phase
    phase = rng.uniform(0, 2 * np.pi)

    # Generate vibrato LFO (low-frequency oscillator)
    lfo = np.sin(2 * np.pi * actual_rate * t + phase)

    # Create onset envelope
    onset_samples = int(onset_delay_sec * sample_rate)
    ramp_samples = int(onset_ramp_sec * sample_rate)

    envelope = np.ones(len(audio))
    envelope[:onset_samples] = 0

    ramp_end = onset_samples + ramp_samples
    if ramp_end < len(audio):
        envelope[onset_samples:ramp_end] = np.linspace(0, 1, ramp_samples)

    # Apply envelope to LFO depth
    modulation_cents = lfo * depth_cents * envelope

    # Convert to pitch ratio
    ratio_curve = np.power(2, modulation_cents / 1200)

    # Apply pitch modulation using block-based shifting
    return _block_pitch_shift(audio, ratio_curve, sample_rate)


def apply_vibrato_natural(
    audio: np.ndarray,
    sample_rate: int,
    rate_hz: float,
    depth_cents: float,
    onset_delay_sec: float = 0.15,
    rate_variation: float = 0.1,
    depth_variation: float = 0.15,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply vibrato with natural rate and depth variations.

    Real singers have slight variations in both vibrato rate and depth
    throughout a sustained note. This function adds slow variations to
    both parameters for a more realistic effect.

    Args:
        audio: Single voice audio [samples]
        sample_rate: Audio sample rate
        rate_hz: Base vibrato frequency
        depth_cents: Base vibrato depth in cents
        onset_delay_sec: Time before vibrato starts
        rate_variation: Proportion of rate variation (0.1 = ±10%)
        depth_variation: Proportion of depth variation (0.15 = ±15%)
        seed: Random seed for reproducibility

    Returns:
        Audio with natural-sounding vibrato
    """
    if depth_cents <= 0 or rate_hz <= 0:
        return audio

    rng = np.random.default_rng(seed)

    duration = len(audio) / sample_rate
    t = np.linspace(0, duration, len(audio))

    # Generate slow modulation curves for rate and depth variation
    # These vary slowly (around 0.5 Hz) to simulate natural inconsistency
    mod_rate = 0.5
    rate_mod = 1.0 + rate_variation * np.sin(
        2 * np.pi * mod_rate * t + rng.uniform(0, 2 * np.pi)
    )
    depth_mod = 1.0 + depth_variation * np.sin(
        2 * np.pi * mod_rate * 0.7 * t + rng.uniform(0, 2 * np.pi)
    )

    # Generate time-varying phase accumulator for variable-rate vibrato
    phase = np.cumsum(2 * np.pi * rate_hz * rate_mod / sample_rate)
    phase += rng.uniform(0, 2 * np.pi)

    # Generate vibrato LFO with varying rate
    lfo = np.sin(phase)

    # Create onset envelope
    onset_samples = int(onset_delay_sec * sample_rate)
    ramp_samples = int(0.1 * sample_rate)  # 100ms ramp

    envelope = np.ones(len(audio))
    envelope[:onset_samples] = 0

    ramp_end = onset_samples + ramp_samples
    if ramp_end < len(audio):
        envelope[onset_samples:ramp_end] = np.linspace(0, 1, ramp_samples)

    # Apply variable depth and envelope
    modulation_cents = lfo * depth_cents * depth_mod * envelope

    # Convert to pitch ratio
    ratio_curve = np.power(2, modulation_cents / 1200)

    # Apply pitch modulation
    return _block_pitch_shift(audio, ratio_curve, sample_rate)


def apply_ensemble_vibrato(
    audio: np.ndarray,
    sample_rate: int,
    rate_hz: float,
    depth_cents: float,
    num_singers: int = 3,
    onset_delay_sec: float = 0.15,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply chorus effect by layering multiple vibratos with different phases.

    When multiple singers vibrato together, their slightly different rates
    and phases create a richer, more complex sound. This function layers
    multiple independent vibratos.

    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        rate_hz: Base vibrato rate
        depth_cents: Base vibrato depth
        num_singers: Number of vibrato layers to blend
        onset_delay_sec: Time before vibrato starts
        seed: Random seed for reproducibility

    Returns:
        Audio with ensemble vibrato effect
    """
    if num_singers <= 1:
        return apply_vibrato(
            audio, sample_rate, rate_hz, depth_cents, onset_delay_sec, seed=seed
        )

    rng = np.random.default_rng(seed)
    layers = []

    for i in range(num_singers):
        # Each singer has slightly different vibrato characteristics
        singer_rate = rate_hz * rng.uniform(0.9, 1.1)
        singer_depth = depth_cents * rng.uniform(0.8, 1.2)
        singer_onset = onset_delay_sec * rng.uniform(0.8, 1.2)

        layer_seed = rng.integers(0, 2**31) if seed is not None else None
        layer = apply_vibrato_natural(
            audio,
            sample_rate,
            singer_rate,
            singer_depth / np.sqrt(num_singers),  # Reduce depth per layer
            singer_onset,
            seed=layer_seed,
        )
        layers.append(layer)

    # Mix layers
    output = np.mean(layers, axis=0)

    return output
