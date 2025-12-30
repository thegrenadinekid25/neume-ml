"""
Main augmentation pipeline orchestrator for choral synthesis.

Combines all effects into a single pipeline that transforms clean
FluidSynth output into realistic choral recordings.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json

from .pitch_scatter import apply_pitch_scatter
from .vibrato import apply_vibrato
from .amplitude_modulation import apply_amplitude_modulation
from .reverb import apply_reverb
from .dynamics import apply_dynamics
from .lowpass import apply_lowpass


# Calibrated ranges from acoustic analysis of reference recordings
# (Lauridsen, Tallis, Whitacre, Bach)
AUGMENTATION_RANGES = {
    "pitch_scatter": {
        "std_cents": (16, 50),  # Per-voice detuning in cents
    },
    "vibrato": {
        "rate_hz": (4.6, 7.0),  # Vibrato frequency
        "depth_cents": (20, 65),  # Vibrato amplitude
        "onset_delay_sec": (0.1, 0.3),  # Time before vibrato starts
    },
    "amplitude_modulation": {
        "depth": (0.39, 0.48),  # Beating effect depth
        "rate_hz": (0.5, 2.0),  # Beat rate from phase drift
    },
    "reverb": {
        "rt60_sec": (1.3, 3.9),  # Decay time
        "wet_dry": (0.39, 0.71),  # Mix ratio
        "pre_delay_ms": (10, 50),  # Initial reflection delay
    },
    "dynamics": {
        "target_lufs": (-33, -20),  # Loudness normalization range
        "crest_factor_db": (15, 24),  # Peak-to-RMS ratio
    },
    "lowpass": {
        "cutoff_hz": (5000, 20000),  # Low-pass filter cutoff (warmer to brighter)
    },
}


@dataclass
class AugmentationConfig:
    """Configuration for the augmentation pipeline."""

    # Pitch scatter
    pitch_scatter_std_cents: float = 30.0

    # Vibrato
    vibrato_rate_hz: float = 5.5
    vibrato_depth_cents: float = 40.0
    vibrato_onset_delay_sec: float = 0.15

    # Amplitude modulation
    am_depth: float = 0.4
    am_rate_hz: float = 1.0

    # Reverb
    reverb_rt60_sec: float = 2.5
    reverb_wet_dry: float = 0.55
    reverb_pre_delay_ms: float = 25.0
    use_convolution_reverb: bool = False
    ir_directory: Optional[str] = None

    # Dynamics
    target_lufs: float = -26.0
    target_crest_factor_db: float = 19.0

    # Low-pass filter (brightness control)
    lowpass_cutoff_hz: float = 12000.0  # Higher = brighter, lower = warmer

    # Random seed for reproducibility
    seed: Optional[int] = None

    @classmethod
    def random(
        cls,
        ranges: Optional[Dict] = None,
        seed: Optional[int] = None,
    ) -> "AugmentationConfig":
        """
        Generate random config within calibrated ranges.

        Args:
            ranges: Custom parameter ranges (defaults to AUGMENTATION_RANGES)
            seed: Random seed for reproducibility

        Returns:
            AugmentationConfig with randomly sampled parameters
        """
        rng = np.random.default_rng(seed)

        if ranges is None:
            ranges = AUGMENTATION_RANGES

        def uniform(key1: str, key2: str) -> float:
            r = ranges[key1][key2]
            return float(rng.uniform(r[0], r[1]))

        return cls(
            pitch_scatter_std_cents=uniform("pitch_scatter", "std_cents"),
            vibrato_rate_hz=uniform("vibrato", "rate_hz"),
            vibrato_depth_cents=uniform("vibrato", "depth_cents"),
            vibrato_onset_delay_sec=uniform("vibrato", "onset_delay_sec"),
            am_depth=uniform("amplitude_modulation", "depth"),
            am_rate_hz=uniform("amplitude_modulation", "rate_hz"),
            reverb_rt60_sec=uniform("reverb", "rt60_sec"),
            reverb_wet_dry=uniform("reverb", "wet_dry"),
            reverb_pre_delay_ms=uniform("reverb", "pre_delay_ms"),
            target_lufs=uniform("dynamics", "target_lufs"),
            target_crest_factor_db=uniform("dynamics", "crest_factor_db"),
            lowpass_cutoff_hz=uniform("lowpass", "cutoff_hz"),
            seed=seed,
        )

    @classmethod
    def from_preset(cls, preset: str) -> "AugmentationConfig":
        """
        Create config from named preset.

        Args:
            preset: One of "minimal", "moderate", "heavy", "cathedral"

        Returns:
            AugmentationConfig with preset values
        """
        presets = {
            "minimal": {
                "pitch_scatter_std_cents": 16,
                "vibrato_rate_hz": 5.0,
                "vibrato_depth_cents": 20,
                "am_depth": 0.39,
                "reverb_rt60_sec": 1.3,
                "reverb_wet_dry": 0.39,
                "lowpass_cutoff_hz": 16000,  # Brighter
            },
            "moderate": {
                "pitch_scatter_std_cents": 33,
                "vibrato_rate_hz": 6.0,
                "vibrato_depth_cents": 40,
                "am_depth": 0.44,
                "reverb_rt60_sec": 2.5,
                "reverb_wet_dry": 0.55,
                "lowpass_cutoff_hz": 10000,  # Balanced
            },
            "heavy": {
                "pitch_scatter_std_cents": 50,
                "vibrato_rate_hz": 7.0,
                "vibrato_depth_cents": 65,
                "am_depth": 0.48,
                "reverb_rt60_sec": 3.9,
                "reverb_wet_dry": 0.71,
                "lowpass_cutoff_hz": 8000,  # Warmer
            },
            "cathedral": {
                "pitch_scatter_std_cents": 40,
                "vibrato_rate_hz": 5.5,
                "vibrato_depth_cents": 45,
                "am_depth": 0.45,
                "reverb_rt60_sec": 3.9,
                "reverb_wet_dry": 0.70,
                "reverb_pre_delay_ms": 45.0,
                "lowpass_cutoff_hz": 7000,  # Warm cathedral sound
            },
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")

        return cls(**presets[preset])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pitch_scatter_std_cents": self.pitch_scatter_std_cents,
            "vibrato_rate_hz": self.vibrato_rate_hz,
            "vibrato_depth_cents": self.vibrato_depth_cents,
            "vibrato_onset_delay_sec": self.vibrato_onset_delay_sec,
            "am_depth": self.am_depth,
            "am_rate_hz": self.am_rate_hz,
            "reverb_rt60_sec": self.reverb_rt60_sec,
            "reverb_wet_dry": self.reverb_wet_dry,
            "reverb_pre_delay_ms": self.reverb_pre_delay_ms,
            "use_convolution_reverb": self.use_convolution_reverb,
            "target_lufs": self.target_lufs,
            "target_crest_factor_db": self.target_crest_factor_db,
            "lowpass_cutoff_hz": self.lowpass_cutoff_hz,
            "seed": self.seed,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AugmentationConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, json_str: str) -> "AugmentationConfig":
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))


class AugmentationPipeline:
    """
    Complete augmentation pipeline for choral audio.

    Takes multi-track (per-voice) or mixed audio and applies
    realistic acoustic effects calibrated from professional recordings.

    The pipeline applies effects in this order:
    1. Per-voice: Pitch scatter (independent per voice)
    2. Per-voice: Vibrato (independent per voice)
    3. Per-voice: Amplitude modulation (beating)
    4. Mix voices together
    5. Post-mix: Reverb
    6. Post-mix: Dynamics normalization
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize the augmentation pipeline.

        Args:
            config: Augmentation parameters. If None, uses random config.
        """
        self.config = config or AugmentationConfig.random()
        self._rng = np.random.default_rng(self.config.seed)

    def process_multitrack(
        self,
        voices: List[np.ndarray],
        sample_rate: int,
    ) -> np.ndarray:
        """
        Process multi-track audio (one array per voice).

        This is the preferred mode - allows per-voice pitch scatter
        and vibrato before mixing, which creates the most realistic
        choral sound.

        Args:
            voices: List of audio arrays, one per voice part
            sample_rate: Audio sample rate

        Returns:
            Augmented and mixed audio
        """
        processed_voices = []

        for i, voice in enumerate(voices):
            # Per-voice processing with independent randomization
            v = voice.copy()
            voice_seed = self._rng.integers(0, 2**31)

            # 1. Pitch scatter (independent per voice)
            v = apply_pitch_scatter(
                v,
                sample_rate,
                std_cents=self.config.pitch_scatter_std_cents,
                seed=voice_seed,
            )

            # 2. Vibrato (independent per voice with slight variation)
            rate_variation = self._rng.uniform(0.9, 1.1)
            depth_variation = self._rng.uniform(0.8, 1.2)
            vibrato_seed = self._rng.integers(0, 2**31)

            v = apply_vibrato(
                v,
                sample_rate,
                rate_hz=self.config.vibrato_rate_hz * rate_variation,
                depth_cents=self.config.vibrato_depth_cents * depth_variation,
                onset_delay_sec=self.config.vibrato_onset_delay_sec,
                seed=vibrato_seed,
            )

            # 3. Amplitude modulation (beating)
            am_seed = self._rng.integers(0, 2**31)
            v = apply_amplitude_modulation(
                v,
                sample_rate,
                depth=self.config.am_depth,
                rate_hz=self.config.am_rate_hz,
                seed=am_seed,
            )

            processed_voices.append(v)

        # Mix voices together
        max_len = max(len(v) for v in processed_voices)
        mixed = np.zeros(max_len)
        for v in processed_voices:
            mixed[: len(v)] += v

        # Normalize mix
        mixed = mixed / len(processed_voices)

        # Apply post-mix processing
        return self._post_mix_processing(mixed, sample_rate)

    def process_mixed(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Process already-mixed audio.

        Less realistic than multitrack mode, but works when
        separate voices aren't available. Effects are applied
        with reduced intensity since they're applied to the mix.

        Args:
            audio: Mixed audio
            sample_rate: Audio sample rate

        Returns:
            Augmented audio
        """
        # Apply pitch scatter to entire mix (reduced intensity)
        audio = apply_pitch_scatter(
            audio,
            sample_rate,
            std_cents=self.config.pitch_scatter_std_cents * 0.5,
            seed=self._rng.integers(0, 2**31),
        )

        # Apply vibrato to mix (reduced intensity)
        audio = apply_vibrato(
            audio,
            sample_rate,
            rate_hz=self.config.vibrato_rate_hz,
            depth_cents=self.config.vibrato_depth_cents * 0.3,
            onset_delay_sec=self.config.vibrato_onset_delay_sec,
            seed=self._rng.integers(0, 2**31),
        )

        # AM still works reasonably on mix
        audio = apply_amplitude_modulation(
            audio,
            sample_rate,
            depth=self.config.am_depth,
            rate_hz=self.config.am_rate_hz,
            seed=self._rng.integers(0, 2**31),
        )

        return self._post_mix_processing(audio, sample_rate)

    def _post_mix_processing(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Apply lowpass, reverb and dynamics to mixed signal."""

        # 4. Low-pass filter (brightness control)
        # Apply before reverb so the reverb tail also gets filtered
        audio = apply_lowpass(
            audio,
            sample_rate,
            cutoff_hz=self.config.lowpass_cutoff_hz,
        )

        # 5. Reverb
        reverb_seed = self._rng.integers(0, 2**31)
        audio = apply_reverb(
            audio,
            sample_rate,
            rt60_sec=self.config.reverb_rt60_sec,
            wet_dry=self.config.reverb_wet_dry,
            pre_delay_ms=self.config.reverb_pre_delay_ms,
            ir_directory=self.config.ir_directory,
            use_convolution=self.config.use_convolution_reverb,
            seed=reverb_seed,
        )

        # 6. Dynamics normalization
        audio = apply_dynamics(
            audio,
            sample_rate,
            target_lufs=self.config.target_lufs,
            target_crest_factor_db=self.config.target_crest_factor_db,
        )

        return audio

    def process_voice_events(
        self,
        voice_events: List[List],
        rendered_voices: List[np.ndarray],
        sample_rate: int,
    ) -> np.ndarray:
        """
        Process voices rendered from voice events.

        This is the intended integration point with the training set generator.

        Args:
            voice_events: Voice event data (for metadata, not used in processing)
            rendered_voices: List of rendered audio arrays, one per voice
            sample_rate: Audio sample rate

        Returns:
            Augmented and mixed audio
        """
        return self.process_multitrack(rendered_voices, sample_rate)


def augment_audio(
    audio: np.ndarray,
    sample_rate: int,
    config: Optional[AugmentationConfig] = None,
) -> tuple:
    """
    Convenience function for single-call augmentation.

    Applies the complete augmentation pipeline to mixed audio
    and returns both the processed audio and the config used.

    Args:
        audio: Input audio (mixed)
        sample_rate: Sample rate
        config: Augmentation config (random if None)

    Returns:
        Tuple of (augmented_audio, config_dict)

    Example:
        >>> audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 88200))
        >>> augmented, config = augment_audio(audio, 44100)
    """
    if config is None:
        config = AugmentationConfig.random()

    pipeline = AugmentationPipeline(config)
    augmented = pipeline.process_mixed(audio, sample_rate)

    return augmented, config.to_dict()


def augment_multitrack(
    voices: List[np.ndarray],
    sample_rate: int,
    config: Optional[AugmentationConfig] = None,
) -> tuple:
    """
    Convenience function for multitrack augmentation.

    Args:
        voices: List of voice audio arrays
        sample_rate: Sample rate
        config: Augmentation config (random if None)

    Returns:
        Tuple of (augmented_audio, config_dict)
    """
    if config is None:
        config = AugmentationConfig.random()

    pipeline = AugmentationPipeline(config)
    augmented = pipeline.process_multitrack(voices, sample_rate)

    return augmented, config.to_dict()
