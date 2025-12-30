"""
Amplitude modulation (beating) augmentation for choral synthesis.

Simulates phase interference between multiple singers on the same part.
Real choral recordings show 39-48% AM depth at 0.5-2 Hz.
"""

import numpy as np
from typing import Optional


def apply_amplitude_modulation(
    audio: np.ndarray,
    sample_rate: int,
    depth: float,
    rate_hz: float,
    num_modulators: int = 3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply slow amplitude modulation to simulate beating.

    When multiple singers sing the same note slightly out of tune,
    their waveforms interfere creating amplitude fluctuations. This
    effect is known as "beating" and is characteristic of ensemble sound.

    The beating is typically slow (0.5-2 Hz) corresponding to the small
    pitch differences between singers (a few cents).

    Args:
        audio: Audio signal
        sample_rate: Audio sample rate
        depth: Modulation depth (0-1, typically 0.39-0.48 for choral)
        rate_hz: Base modulation rate (0.5-2 Hz)
        num_modulators: Number of independent AM sources to sum
        seed: Random seed for reproducibility

    Returns:
        Amplitude-modulated audio

    Example:
        >>> audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 88200))
        >>> modulated = apply_amplitude_modulation(audio, 44100, depth=0.4, rate_hz=1.0)
    """
    if depth <= 0:
        return audio

    rng = np.random.default_rng(seed)

    duration = len(audio) / sample_rate
    t = np.linspace(0, duration, len(audio))

    # Sum multiple slow modulators with random phases and rates
    # This creates a more complex, realistic beating pattern
    modulation = np.ones(len(audio))

    for i in range(num_modulators):
        # Each modulator has slightly different rate
        mod_rate = rate_hz * rng.uniform(0.8, 1.2)
        phase = rng.uniform(0, 2 * np.pi)
        mod_depth = depth / num_modulators

        modulation += mod_depth * np.sin(2 * np.pi * mod_rate * t + phase)

    # Normalize to prevent clipping while maintaining relative dynamics
    modulation = modulation / np.max(np.abs(modulation))

    return audio * modulation


def apply_beating(
    audio: np.ndarray,
    sample_rate: int,
    depth: float,
    beat_rate_hz: float,
    irregularity: float = 0.2,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply realistic beating with irregular modulation.

    Real beating from pitch interference is not perfectly sinusoidal.
    This function adds some irregularity to the modulation pattern.

    Args:
        audio: Audio signal
        sample_rate: Sample rate
        depth: Maximum modulation depth (0-1)
        beat_rate_hz: Central beat frequency
        irregularity: Amount of random variation (0-1)
        seed: Random seed for reproducibility

    Returns:
        Audio with beating effect
    """
    if depth <= 0:
        return audio

    rng = np.random.default_rng(seed)

    duration = len(audio) / sample_rate
    t = np.linspace(0, duration, len(audio))

    # Generate base sinusoidal modulation
    phase = rng.uniform(0, 2 * np.pi)
    base_mod = np.sin(2 * np.pi * beat_rate_hz * t + phase)

    # Add irregularity via low-frequency noise
    if irregularity > 0:
        # Generate smooth random modulation
        num_points = max(10, int(duration * 5))
        random_points = rng.uniform(-1, 1, num_points)
        t_points = np.linspace(0, duration, num_points)
        irregular_component = np.interp(t, t_points, random_points)

        # Smooth it
        from scipy.ndimage import gaussian_filter1d
        sigma = sample_rate * 0.1  # 100ms smoothing
        irregular_component = gaussian_filter1d(irregular_component, sigma=sigma)
        irregular_component = irregular_component / np.max(np.abs(irregular_component) + 1e-8)

        # Blend regular and irregular components
        modulation = (1 - irregularity) * base_mod + irregularity * irregular_component
    else:
        modulation = base_mod

    # Scale and offset to create amplitude envelope
    # Modulation goes from (1-depth) to 1
    envelope = 1 - depth * 0.5 + depth * 0.5 * modulation

    return audio * envelope


def apply_chorus_am(
    audio: np.ndarray,
    sample_rate: int,
    depth: float,
    num_voices: int = 4,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply amplitude modulation simulating chorus interference patterns.

    Models the complex beating patterns that emerge when multiple
    singers are slightly detuned from each other. Uses multiple
    independent modulators at different rates.

    Args:
        audio: Audio signal
        sample_rate: Sample rate
        depth: Overall modulation depth
        num_voices: Number of simulated voices (affects complexity)
        seed: Random seed for reproducibility

    Returns:
        Audio with chorus-like AM
    """
    if depth <= 0:
        return audio

    rng = np.random.default_rng(seed)

    duration = len(audio) / sample_rate
    t = np.linspace(0, duration, len(audio))

    # Generate multiple beat frequencies
    # Pitch differences of a few cents create beat rates of ~1 Hz
    base_beat_rate = 1.0
    modulation = np.ones(len(audio))

    for i in range(num_voices):
        # Each voice-pair interaction has a different beat rate
        beat_rate = base_beat_rate * rng.uniform(0.3, 2.5)
        phase = rng.uniform(0, 2 * np.pi)
        voice_depth = depth / np.sqrt(num_voices)  # Scale depth

        # Add second harmonic for more realism
        mod = voice_depth * np.sin(2 * np.pi * beat_rate * t + phase)
        mod += voice_depth * 0.3 * np.sin(2 * np.pi * beat_rate * 2 * t + phase * 1.5)

        modulation += mod

    # Normalize
    modulation = modulation / np.max(np.abs(modulation))

    # Ensure modulation stays positive (amplitude multiplier)
    modulation = np.maximum(modulation, 0.1)

    return audio * modulation


def apply_tremolo(
    audio: np.ndarray,
    sample_rate: int,
    depth: float,
    rate_hz: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply simple tremolo (regular amplitude modulation).

    Unlike beating which is caused by pitch interference, tremolo
    is intentional rapid volume variation. This can be used to
    simulate expressive singing techniques.

    Args:
        audio: Audio signal
        sample_rate: Sample rate
        depth: Modulation depth (0-1)
        rate_hz: Tremolo rate (typically 4-8 Hz)
        seed: Random seed for reproducibility

    Returns:
        Audio with tremolo effect
    """
    if depth <= 0:
        return audio

    rng = np.random.default_rng(seed)

    duration = len(audio) / sample_rate
    t = np.linspace(0, duration, len(audio))

    phase = rng.uniform(0, 2 * np.pi)
    modulation = 1 - depth * 0.5 + depth * 0.5 * np.sin(2 * np.pi * rate_hz * t + phase)

    return audio * modulation
