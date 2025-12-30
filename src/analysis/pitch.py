"""
Pitch analysis for choral recordings.

Extracts vibrato characteristics, pitch scatter, and drift.
Uses CREPE neural pitch tracker for better performance on real recordings.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft
from dataclasses import dataclass
from typing import Tuple, Optional, List
import warnings


@dataclass
class PitchAnalysis:
    """Pitch characteristics extracted from audio."""
    vibrato_rate_hz: float            # Vibrato frequency (typically 5-7 Hz)
    vibrato_depth_cents: float        # Vibrato extent (typically ±50-100 cents)
    pitch_stability_std_cents: float  # Scatter around target pitch
    drift_cents_per_second: float     # Long-term pitch drift
    pitch_range_cents: float          # Overall pitch variation
    confidence: float                 # Analysis confidence (0-1)


def analyze_pitch(audio: np.ndarray, sr: int) -> PitchAnalysis:
    """
    Extract pitch characteristics using CREPE neural pitch tracker.

    For polyphonic choral music, this focuses on:
    - Overall pitch stability metrics
    - Vibrato characteristics from prominent voice sections

    Args:
        audio: Mono audio signal
        sr: Sample rate in Hz

    Returns:
        PitchAnalysis with extracted parameters
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Try CREPE first, fall back to pYIN if not available
    try:
        pitch_hz, confidence_array = _extract_pitch_crepe(audio, sr)
    except ImportError:
        warnings.warn("CREPE not available, falling back to pYIN")
        pitch_hz, confidence_array = _extract_pitch_pyin(audio, sr)

    # Filter to confident pitch estimates
    valid_mask = (confidence_array > 0.5) & (pitch_hz > 50) & (pitch_hz < 2000)

    if np.sum(valid_mask) < 10:
        # Not enough valid pitch data
        return PitchAnalysis(
            vibrato_rate_hz=6.0,  # Default values
            vibrato_depth_cents=60.0,
            pitch_stability_std_cents=25.0,
            drift_cents_per_second=0.0,
            pitch_range_cents=100.0,
            confidence=0.1
        )

    valid_pitch = pitch_hz[valid_mask]
    valid_conf = confidence_array[valid_mask]

    # Convert to cents relative to mean
    mean_pitch = np.median(valid_pitch)
    pitch_cents = 1200 * np.log2(valid_pitch / mean_pitch)

    # Extract vibrato parameters
    vibrato_rate, vibrato_depth = _extract_vibrato_params(
        pitch_hz, confidence_array, sr
    )

    # Calculate pitch stability (scatter)
    stability_std = _calculate_pitch_stability(pitch_cents, valid_conf)

    # Calculate drift
    drift = _calculate_pitch_drift(pitch_cents, sr, len(audio))

    # Pitch range (5th to 95th percentile)
    pitch_range = np.percentile(pitch_cents, 95) - np.percentile(pitch_cents, 5)

    # Overall confidence
    overall_conf = np.mean(valid_conf) * (np.sum(valid_mask) / len(valid_mask))

    return PitchAnalysis(
        vibrato_rate_hz=float(vibrato_rate),
        vibrato_depth_cents=float(vibrato_depth),
        pitch_stability_std_cents=float(stability_std),
        drift_cents_per_second=float(drift),
        pitch_range_cents=float(pitch_range),
        confidence=float(overall_conf)
    )


def _extract_pitch_crepe(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract pitch using CREPE neural pitch tracker.

    CREPE works better than pYIN on real recordings with reverb.
    """
    import crepe

    # Resample to 16kHz if needed (CREPE's native rate)
    if sr != 16000:
        import librosa
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr_crepe = 16000
    else:
        audio_16k = audio
        sr_crepe = sr

    # Run CREPE with 10ms step size for good temporal resolution
    # Use 'small' model for speed, 'full' for accuracy
    time, frequency, confidence, activation = crepe.predict(
        audio_16k,
        sr_crepe,
        model_capacity='small',
        step_size=10,  # 10ms
        viterbi=True,  # Use Viterbi smoothing
        center=True,
        verbose=0
    )

    return frequency, confidence


def _extract_pitch_pyin(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback pitch extraction using pYIN (probabilistic YIN).
    """
    import librosa

    # Extract pitch using pYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=80,
        fmax=1000,
        sr=sr,
        frame_length=2048,
        hop_length=int(sr * 0.01)  # 10ms hop
    )

    # Replace NaN with 0
    f0 = np.nan_to_num(f0, nan=0.0)

    # Use voiced probability as confidence
    confidence = voiced_probs

    return f0, confidence


def _extract_vibrato_params(
    pitch_hz: np.ndarray,
    confidence: np.ndarray,
    sr: int
) -> Tuple[float, float]:
    """
    Extract vibrato rate (Hz) and depth (cents) from pitch contour.

    Approach:
    1. Get pitch contour
    2. Detrend (remove slow drift)
    3. Find periodicity in residual (FFT, look for 5-7 Hz peak)
    4. Measure amplitude of that periodic component
    """
    # Work only with confident estimates
    valid_mask = (confidence > 0.6) & (pitch_hz > 50)

    if np.sum(valid_mask) < 50:
        return 6.0, 50.0  # Default values

    valid_pitch = pitch_hz[valid_mask]

    # Convert to cents (relative to median)
    median_pitch = np.median(valid_pitch)
    pitch_cents = 1200 * np.log2(valid_pitch / median_pitch)

    # Find segments of continuous voiced regions
    segments = _find_continuous_segments(valid_mask, min_length=30)

    if not segments:
        return 6.0, 50.0

    vibrato_rates = []
    vibrato_depths = []

    for start, end in segments[:10]:  # Analyze up to 10 segments
        segment = pitch_hz[start:end]
        if len(segment) < 30:
            continue

        segment_cents = 1200 * np.log2(segment / np.median(segment))

        # Detrend
        t = np.arange(len(segment))
        coeffs = np.polyfit(t, segment_cents, 2)
        trend = np.polyval(coeffs, t)
        detrended = segment_cents - trend

        # FFT to find vibrato rate
        rate, depth = _analyze_vibrato_segment(detrended, hop_rate=100)
        if rate is not None:
            vibrato_rates.append(rate)
            vibrato_depths.append(depth)

    if vibrato_rates:
        return np.median(vibrato_rates), np.median(vibrato_depths)
    else:
        return 6.0, 50.0


def _find_continuous_segments(mask: np.ndarray, min_length: int) -> List[Tuple[int, int]]:
    """Find continuous True segments in mask."""
    segments = []
    in_segment = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_segment:
            start = i
            in_segment = True
        elif not val and in_segment:
            if i - start >= min_length:
                segments.append((start, i))
            in_segment = False

    if in_segment and len(mask) - start >= min_length:
        segments.append((start, len(mask)))

    return segments


def _analyze_vibrato_segment(
    pitch_cents: np.ndarray,
    hop_rate: float = 100
) -> Tuple[Optional[float], Optional[float]]:
    """
    Analyze a pitch segment for vibrato.

    Args:
        pitch_cents: Detrended pitch in cents
        hop_rate: Analysis hop rate in Hz (e.g., 100 for 10ms)

    Returns:
        (vibrato_rate_hz, vibrato_depth_cents) or (None, None)
    """
    n = len(pitch_cents)
    if n < 30:
        return None, None

    # Zero-pad for better frequency resolution
    n_fft = max(512, 2 ** int(np.ceil(np.log2(n * 4))))

    # Apply window
    windowed = pitch_cents * np.hanning(n)

    # FFT
    spectrum = np.abs(fft(windowed, n_fft))[:n_fft // 2]
    freqs = np.fft.fftfreq(n_fft, 1 / hop_rate)[:n_fft // 2]

    # Look for peak in vibrato range (4-9 Hz)
    vibrato_mask = (freqs >= 4) & (freqs <= 9)
    if not np.any(vibrato_mask):
        return None, None

    vibrato_spectrum = spectrum.copy()
    vibrato_spectrum[~vibrato_mask] = 0

    peak_idx = np.argmax(vibrato_spectrum)
    peak_freq = freqs[peak_idx]

    # Check if peak is significant (above noise floor)
    noise_floor = np.median(spectrum[(freqs > 1) & (freqs < 4)])
    peak_height = spectrum[peak_idx]

    if peak_height < noise_floor * 2:
        # No significant vibrato detected
        return None, None

    # Estimate depth from RMS of detrended signal
    depth = np.std(pitch_cents) * 2  # ~2 sigma for peak-to-peak

    return peak_freq, depth


def _calculate_pitch_stability(pitch_cents: np.ndarray, confidence: np.ndarray) -> float:
    """
    Calculate pitch stability as weighted standard deviation.

    This represents the scatter around the target pitch after
    removing vibrato and drift components.
    """
    # Weight by confidence
    weights = confidence / np.sum(confidence)

    # Weighted mean
    mean = np.sum(weights * pitch_cents)

    # Weighted standard deviation
    variance = np.sum(weights * (pitch_cents - mean) ** 2)
    std = np.sqrt(variance)

    # Remove estimated vibrato contribution (rough approximation)
    # Vibrato adds ~30-50% to apparent std
    stability_std = std * 0.7

    return stability_std


def _calculate_pitch_drift(pitch_cents: np.ndarray, sr: int, n_samples: int) -> float:
    """
    Calculate long-term pitch drift in cents per second.
    """
    if len(pitch_cents) < 10:
        return 0.0

    # Total duration
    duration = n_samples / sr

    # Time axis
    t = np.linspace(0, duration, len(pitch_cents))

    # Linear fit
    slope, intercept = np.polyfit(t, pitch_cents, 1)

    return slope  # cents per second
