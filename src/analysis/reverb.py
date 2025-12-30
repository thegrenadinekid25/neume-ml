"""
Reverb analysis for choral recordings.

Estimates reverb characteristics from audio without impulse responses.
Uses energy decay analysis after detected transients.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass
from typing import Optional, Tuple, List
import librosa


@dataclass
class ReverbAnalysis:
    """Reverb characteristics extracted from audio."""
    rt60_estimate: float       # Estimated RT60 in seconds
    early_decay_time: float    # EDT in seconds (decay from 0 to -10dB)
    clarity_c50: float         # Energy ratio before/after 50ms (dB)
    wet_dry_estimate: float    # Estimated wet/dry ratio (0-1)
    confidence: float          # Confidence in the estimate (0-1)


def analyze_reverb(audio: np.ndarray, sr: int) -> ReverbAnalysis:
    """
    Estimate reverb characteristics from audio.

    Approach:
    1. Find transient-like onsets (consonants, note attacks)
    2. Analyze energy decay after these points
    3. Fit exponential decay to estimate RT60

    For choral music without clear transients, uses:
    - Spectral flux for onset detection
    - Analyzes decay in frequency bands separately
    - Takes median across multiple decay measurements

    Args:
        audio: Mono audio signal
        sr: Sample rate in Hz

    Returns:
        ReverbAnalysis with estimated parameters
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Find onset candidates using spectral flux
    onsets = _find_onset_candidates(audio, sr)

    if len(onsets) < 3:
        # Fallback: use energy envelope analysis
        return _analyze_reverb_from_envelope(audio, sr)

    # Analyze decay after each onset
    rt60_estimates = []
    edt_estimates = []

    for onset_sample in onsets[:20]:  # Limit to first 20 for efficiency
        decay_curve = _extract_decay_curve(audio, onset_sample, sr)
        if decay_curve is not None:
            rt60 = _estimate_rt60_from_decay(decay_curve, sr)
            edt = _estimate_edt_from_decay(decay_curve, sr)
            if rt60 is not None and 0.3 < rt60 < 15.0:  # Plausible range
                rt60_estimates.append(rt60)
            if edt is not None and 0.1 < edt < 10.0:
                edt_estimates.append(edt)

    if not rt60_estimates:
        return _analyze_reverb_from_envelope(audio, sr)

    # Use median for robustness
    rt60_median = np.median(rt60_estimates)
    edt_median = np.median(edt_estimates) if edt_estimates else rt60_median * 0.5

    # Confidence based on consistency of estimates
    if len(rt60_estimates) >= 3:
        cv = np.std(rt60_estimates) / np.mean(rt60_estimates)  # Coefficient of variation
        confidence = max(0.0, min(1.0, 1.0 - cv))
    else:
        confidence = 0.3

    # Estimate clarity C50 and wet/dry
    c50 = _estimate_clarity_c50(audio, onsets, sr)
    wet_dry = _estimate_wet_dry(rt60_median, c50)

    return ReverbAnalysis(
        rt60_estimate=float(rt60_median),
        early_decay_time=float(edt_median),
        clarity_c50=float(c50),
        wet_dry_estimate=float(wet_dry),
        confidence=float(confidence)
    )


def _find_onset_candidates(audio: np.ndarray, sr: int) -> List[int]:
    """
    Find onset candidates using spectral flux.

    Returns sample indices of detected onsets.
    """
    # Compute onset strength envelope
    hop_length = int(sr * 0.01)  # 10ms hop
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)

    # Peak picking with adaptive threshold
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=False,
        pre_max=3,
        post_max=3,
        pre_avg=10,
        post_avg=10,
        delta=0.2,
        wait=20
    )

    # Convert frames to samples
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)

    # Filter to keep only strong onsets (above median strength)
    if len(onset_frames) > 0:
        strengths = onset_env[onset_frames]
        threshold = np.median(strengths)
        strong_mask = strengths >= threshold
        onset_samples = onset_samples[strong_mask]

    return onset_samples.tolist()


def _extract_decay_curve(audio: np.ndarray, onset_sample: int, sr: int) -> Optional[np.ndarray]:
    """
    Extract energy decay curve after an onset.

    Returns smoothed energy envelope for decay analysis.
    """
    # Window: 50ms before to 2s after onset
    pre_samples = int(sr * 0.05)
    post_samples = int(sr * 2.0)

    start = max(0, onset_sample - pre_samples)
    end = min(len(audio), onset_sample + post_samples)

    if end - start < int(sr * 0.5):  # Need at least 500ms
        return None

    segment = audio[start:end]

    # Compute energy envelope using Hilbert transform
    analytic_signal = signal.hilbert(segment)
    envelope = np.abs(analytic_signal)

    # Smooth the envelope
    window_size = int(sr * 0.02)  # 20ms smoothing
    envelope = uniform_filter1d(envelope, size=window_size)

    # Find the peak (should be near the onset)
    peak_region = int(sr * 0.1)  # Look within 100ms
    peak_idx = np.argmax(envelope[:peak_region])

    # Return decay from peak onwards
    decay = envelope[peak_idx:]

    if len(decay) < int(sr * 0.3):  # Need at least 300ms of decay
        return None

    return decay


def _estimate_rt60_from_decay(decay_curve: np.ndarray, sr: int) -> Optional[float]:
    """
    Estimate RT60 from decay curve using Schroeder backward integration.

    Schroeder integration is more robust than direct curve fitting.
    """
    # Normalize
    decay_curve = decay_curve / (np.max(decay_curve) + 1e-10)

    # Square for energy
    energy = decay_curve ** 2

    # Schroeder backward integration
    schroeder = np.cumsum(energy[::-1])[::-1]
    schroeder = schroeder / (schroeder[0] + 1e-10)

    # Convert to dB
    schroeder_db = 10 * np.log10(schroeder + 1e-10)

    # Find the region from -5dB to -35dB for fitting
    # (This range is standard for RT60 estimation from T30)
    try:
        idx_5db = np.where(schroeder_db <= -5)[0][0]
        idx_35db = np.where(schroeder_db <= -35)[0][0]
    except IndexError:
        # Decay doesn't reach -35dB, try smaller range
        try:
            idx_5db = np.where(schroeder_db <= -5)[0][0]
            idx_25db = np.where(schroeder_db <= -25)[0][0]
            idx_35db = idx_25db
        except IndexError:
            return None

    if idx_35db <= idx_5db:
        return None

    # Linear fit in dB domain
    x = np.arange(idx_5db, idx_35db) / sr
    y = schroeder_db[idx_5db:idx_35db]

    if len(x) < 3:
        return None

    # Fit line: y = slope * x + intercept
    slope, intercept = np.polyfit(x, y, 1)

    if slope >= 0:  # Not decaying
        return None

    # RT60 = time to decay 60dB
    rt60 = -60 / slope

    return rt60


def _estimate_edt_from_decay(decay_curve: np.ndarray, sr: int) -> Optional[float]:
    """
    Estimate Early Decay Time (EDT) from decay curve.

    EDT is defined as time for first 10dB of decay, extrapolated to 60dB.
    """
    decay_curve = decay_curve / (np.max(decay_curve) + 1e-10)
    energy = decay_curve ** 2

    schroeder = np.cumsum(energy[::-1])[::-1]
    schroeder = schroeder / (schroeder[0] + 1e-10)
    schroeder_db = 10 * np.log10(schroeder + 1e-10)

    # Find region from 0dB to -10dB
    try:
        idx_10db = np.where(schroeder_db <= -10)[0][0]
    except IndexError:
        return None

    if idx_10db < 3:
        return None

    x = np.arange(0, idx_10db) / sr
    y = schroeder_db[0:idx_10db]

    slope, intercept = np.polyfit(x, y, 1)

    if slope >= 0:
        return None

    # EDT extrapolated to 60dB
    edt = -60 / slope

    return edt


def _analyze_reverb_from_envelope(audio: np.ndarray, sr: int) -> ReverbAnalysis:
    """
    Fallback reverb analysis using overall energy envelope.

    Less accurate but works when clear transients aren't available.
    """
    # Compute RMS envelope with longer window
    frame_length = int(sr * 0.05)
    hop_length = int(sr * 0.01)

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    # Find segments where energy drops significantly
    rms_db = 20 * np.log10(rms + 1e-10)

    # Look for rapid drops followed by decay
    diff = np.diff(rms_db)
    drops = np.where(diff < -2)[0]  # Significant drops

    rt60_estimates = []

    for drop_idx in drops[:10]:
        # Analyze decay after this drop
        if drop_idx + int(sr * 1.0 / hop_length) > len(rms):
            continue

        decay_segment = rms[drop_idx:drop_idx + int(sr * 2.0 / hop_length)]
        if len(decay_segment) < int(sr * 0.3 / hop_length):
            continue

        decay_db = 20 * np.log10(decay_segment + 1e-10)
        decay_db = decay_db - decay_db[0]  # Normalize to 0dB start

        # Simple exponential fit
        t = np.arange(len(decay_db)) * hop_length / sr

        try:
            # Find time to -20dB
            idx_20db = np.where(decay_db <= -20)[0]
            if len(idx_20db) > 0:
                t_20db = t[idx_20db[0]]
                rt60_est = t_20db * 3  # Extrapolate to -60dB
                if 0.5 < rt60_est < 12.0:
                    rt60_estimates.append(rt60_est)
        except:
            continue

    if rt60_estimates:
        rt60 = np.median(rt60_estimates)
        confidence = 0.4  # Lower confidence for envelope method
    else:
        rt60 = 2.0  # Default fallback
        confidence = 0.1

    return ReverbAnalysis(
        rt60_estimate=float(rt60),
        early_decay_time=float(rt60 * 0.5),  # Rough estimate
        clarity_c50=float(-3.0),  # Typical value
        wet_dry_estimate=float(0.3),  # Conservative estimate
        confidence=float(confidence)
    )


def _estimate_clarity_c50(audio: np.ndarray, onsets: List[int], sr: int) -> float:
    """
    Estimate clarity C50 (ratio of energy before/after 50ms).

    Higher C50 = clearer/drier sound.
    """
    if not onsets:
        return 0.0

    c50_values = []
    samples_50ms = int(sr * 0.05)

    for onset in onsets[:10]:
        if onset + int(sr * 0.5) > len(audio):
            continue

        # Energy in first 50ms
        early_energy = np.sum(audio[onset:onset + samples_50ms] ** 2)

        # Energy from 50ms to 500ms
        late_energy = np.sum(audio[onset + samples_50ms:onset + int(sr * 0.5)] ** 2)

        if late_energy > 0:
            c50 = 10 * np.log10((early_energy + 1e-10) / (late_energy + 1e-10))
            c50_values.append(c50)

    return np.median(c50_values) if c50_values else 0.0


def _estimate_wet_dry(rt60: float, c50: float) -> float:
    """
    Estimate wet/dry ratio from RT60 and C50.

    This is a heuristic mapping, not a precise measurement.
    """
    # Longer RT60 = more wet
    # Lower C50 = more wet

    # Normalize RT60 to 0-1 range (assuming 0.5-8s range)
    rt60_factor = np.clip((rt60 - 0.5) / 7.5, 0, 1)

    # Normalize C50 (typical range -10 to +10 dB)
    c50_factor = np.clip((10 - c50) / 20, 0, 1)

    # Combine (weight RT60 more heavily)
    wet_dry = 0.6 * rt60_factor + 0.4 * c50_factor

    return float(np.clip(wet_dry, 0, 1))
