"""
Reverb augmentation for choral synthesis.

Applies convolution or algorithmic reverb matching cathedral/hall acoustics.
Real choral recordings show RT60 of 1.3-3.9s with 39-71% wet mix.
"""

import numpy as np
from scipy.signal import fftconvolve
from typing import Optional, Dict, Tuple
import os
from pathlib import Path


class ReverbProcessor:
    """
    Reverb processor with both convolution and algorithmic modes.

    Convolution mode uses impulse responses for realistic spaces.
    Algorithmic mode generates synthetic reverb for variety.
    """

    def __init__(self, ir_directory: Optional[str] = None):
        """
        Initialize the reverb processor.

        Args:
            ir_directory: Path to directory containing impulse response files (.wav)
        """
        self.ir_cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self.ir_directory = ir_directory

        if ir_directory and os.path.exists(ir_directory):
            self._load_impulse_responses()

    def _load_impulse_responses(self) -> None:
        """Load all IR files from directory into cache."""
        try:
            import soundfile as sf
        except ImportError:
            return

        ir_path = Path(self.ir_directory)
        for filename in ir_path.glob("*.wav"):
            try:
                ir, sr = sf.read(str(filename))
                if ir.ndim > 1:
                    ir = ir.mean(axis=1)  # Convert to mono
                # Normalize IR
                ir = ir / (np.max(np.abs(ir)) + 1e-10)
                self.ir_cache[filename.name] = (ir, sr)
            except Exception:
                continue

    def apply_convolution(
        self,
        audio: np.ndarray,
        sample_rate: int,
        wet_dry: float,
        ir_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply convolution reverb using an impulse response.

        Args:
            audio: Input audio
            sample_rate: Audio sample rate
            wet_dry: Wet/dry mix (0=dry, 1=wet)
            ir_name: Specific IR filename to use, or None for random
            seed: Random seed for IR selection

        Returns:
            Reverb-processed audio
        """
        if not self.ir_cache:
            # Fall back to algorithmic
            return self.apply_algorithmic(audio, sample_rate, 2.0, wet_dry)

        rng = np.random.default_rng(seed)

        if ir_name is None:
            ir_name = rng.choice(list(self.ir_cache.keys()))

        if ir_name not in self.ir_cache:
            ir_name = list(self.ir_cache.keys())[0]

        ir, ir_sr = self.ir_cache[ir_name]

        # Resample IR if sample rates don't match
        if ir_sr != sample_rate:
            from scipy.signal import resample
            new_length = int(len(ir) * sample_rate / ir_sr)
            ir = resample(ir, new_length)

        # Convolve
        wet = fftconvolve(audio, ir, mode="full")[: len(audio)]

        # Normalize wet signal to match dry signal level
        if np.max(np.abs(wet)) > 0:
            wet = wet / np.max(np.abs(wet)) * np.max(np.abs(audio))

        # Mix dry and wet
        return (1 - wet_dry) * audio + wet_dry * wet

    def apply_algorithmic(
        self,
        audio: np.ndarray,
        sample_rate: int,
        rt60_sec: float,
        wet_dry: float,
        pre_delay_ms: float = 20,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply algorithmic reverb (Schroeder-style).

        Generates a synthetic impulse response and convolves with input.
        Good for variety when real IRs are limited.

        Args:
            audio: Input audio
            sample_rate: Audio sample rate
            rt60_sec: Reverb decay time (time to -60dB)
            wet_dry: Wet/dry mix
            pre_delay_ms: Initial delay in milliseconds
            seed: Random seed for reproducibility

        Returns:
            Reverb-processed audio
        """
        # Generate synthetic impulse response
        ir = self._generate_ir(sample_rate, rt60_sec, pre_delay_ms, seed)

        # Convolve
        wet = fftconvolve(audio, ir, mode="full")[: len(audio)]

        # Normalize
        if np.max(np.abs(wet)) > 0:
            wet = wet / np.max(np.abs(wet)) * np.max(np.abs(audio))

        return (1 - wet_dry) * audio + wet_dry * wet

    def _generate_ir(
        self,
        sample_rate: int,
        rt60_sec: float,
        pre_delay_ms: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate synthetic impulse response.

        Creates a noise-based IR with exponential decay and early reflections.
        """
        rng = np.random.default_rng(seed)

        # IR length based on RT60 (extend beyond RT60 for full decay)
        ir_length = int(rt60_sec * sample_rate * 1.5)

        # Pre-delay samples
        pre_delay_samples = int(pre_delay_ms * sample_rate / 1000)

        # Generate noise burst with exponential decay
        t = np.arange(ir_length) / sample_rate
        # -60dB decay at rt60 means decay constant = -6.91/rt60
        decay = np.exp(-6.91 * t / rt60_sec)

        noise = rng.standard_normal(ir_length)
        ir = noise * decay

        # Add early reflections (simplified room simulation)
        early_reflections = [
            (10, 0.8),   # 10ms, 80% amplitude
            (17, 0.6),   # 17ms, 60%
            (23, 0.5),   # 23ms, 50%
            (31, 0.4),   # 31ms, 40%
            (43, 0.3),   # 43ms, 30%
            (57, 0.25),  # 57ms, 25%
        ]

        for delay_ms, gain in early_reflections:
            delay_samples = int(delay_ms * sample_rate / 1000)
            if delay_samples < len(ir):
                ir[delay_samples] += gain * rng.choice([-1, 1])

        # Add pre-delay
        ir = np.concatenate([np.zeros(pre_delay_samples), ir])

        # Normalize
        ir = ir / (np.max(np.abs(ir)) + 1e-10)

        return ir


def apply_reverb(
    audio: np.ndarray,
    sample_rate: int,
    rt60_sec: float,
    wet_dry: float,
    pre_delay_ms: float = 20,
    ir_directory: Optional[str] = None,
    use_convolution: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function for reverb application.

    Args:
        audio: Input audio
        sample_rate: Audio sample rate
        rt60_sec: Target RT60 (used for algorithmic, ignored for convolution)
        wet_dry: Wet/dry mix (0.39-0.71 typical for choral)
        pre_delay_ms: Pre-delay in ms (10-50 typical)
        ir_directory: Path to impulse responses
        use_convolution: Whether to use convolution mode
        seed: Random seed for reproducibility

    Returns:
        Reverb-processed audio

    Example:
        >>> audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> reverbed = apply_reverb(audio, 44100, rt60_sec=2.5, wet_dry=0.55)
    """
    processor = ReverbProcessor(ir_directory)

    if use_convolution and processor.ir_cache:
        return processor.apply_convolution(audio, sample_rate, wet_dry, seed=seed)
    else:
        return processor.apply_algorithmic(
            audio, sample_rate, rt60_sec, wet_dry, pre_delay_ms, seed=seed
        )


def apply_room_simulation(
    audio: np.ndarray,
    sample_rate: int,
    room_size: str = "cathedral",
    wet_dry: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply reverb based on room type presets.

    Provides easy-to-use presets for common choral recording spaces.

    Args:
        audio: Input audio
        sample_rate: Sample rate
        room_size: One of "small", "medium", "large", "cathedral"
        wet_dry: Wet/dry mix
        seed: Random seed for reproducibility

    Returns:
        Reverb-processed audio
    """
    room_params = {
        "small": {"rt60": 0.8, "pre_delay": 5},
        "medium": {"rt60": 1.5, "pre_delay": 15},
        "large": {"rt60": 2.5, "pre_delay": 30},
        "cathedral": {"rt60": 3.5, "pre_delay": 45},
    }

    params = room_params.get(room_size, room_params["large"])

    return apply_reverb(
        audio,
        sample_rate,
        rt60_sec=params["rt60"],
        wet_dry=wet_dry,
        pre_delay_ms=params["pre_delay"],
        seed=seed,
    )


def apply_stereo_reverb(
    audio_left: np.ndarray,
    audio_right: np.ndarray,
    sample_rate: int,
    rt60_sec: float,
    wet_dry: float,
    stereo_width: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply stereo reverb with decorrelated left/right channels.

    Creates a wider, more spacious reverb by using different random
    seeds for left and right channels.

    Args:
        audio_left: Left channel input
        audio_right: Right channel input
        sample_rate: Sample rate
        rt60_sec: Reverb decay time
        wet_dry: Wet/dry mix
        stereo_width: Amount of channel decorrelation (0-1)
        seed: Base random seed

    Returns:
        Tuple of (left_output, right_output)
    """
    rng = np.random.default_rng(seed)

    # Generate different seeds for L/R channels
    seed_l = rng.integers(0, 2**31) if seed is not None else None
    seed_r = rng.integers(0, 2**31) if seed is not None else None

    # Process left channel
    left_out = apply_reverb(audio_left, sample_rate, rt60_sec, wet_dry, seed=seed_l)

    # Process right channel with different random pattern
    right_out = apply_reverb(audio_right, sample_rate, rt60_sec, wet_dry, seed=seed_r)

    # Blend channels based on stereo width
    # Full width = completely different, 0 = mono reverb
    if stereo_width < 1.0:
        mono_reverb = (left_out + right_out) / 2
        left_out = stereo_width * left_out + (1 - stereo_width) * mono_reverb
        right_out = stereo_width * right_out + (1 - stereo_width) * mono_reverb

    return left_out, right_out
