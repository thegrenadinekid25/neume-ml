"""
Spectral analysis for choral recordings.

Extracts formant characteristics, harmonic-to-noise ratio, and spectral features.
Uses Praat via parselmouth for formant analysis (gold standard).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings


@dataclass
class SpectralAnalysis:
    """Spectral characteristics extracted from audio."""
    formant_freqs: List[float]       # F1, F2, F3 average frequencies (Hz)
    formant_bandwidths: List[float]  # F1, F2, F3 bandwidths (Hz)
    spectral_centroid_mean: float    # Mean spectral centroid (Hz)
    spectral_centroid_std: float     # Centroid variability (Hz)
    harmonic_to_noise_ratio: float   # HNR in dB (higher = cleaner)
    spectral_flux_mean: float        # Rate of spectral change
    spectral_rolloff_mean: float     # Frequency below which 85% of energy lies
    # Chorus width / pitch scatter metrics
    harmonic_peak_width_cents: float # Mean width of harmonic peaks (wider = more chorus)
    inharmonicity: float             # Deviation from perfect harmonic series (0-1)
    amplitude_modulation_depth: float # AM depth from beating (0-1, higher = more scatter)
    estimated_pitch_scatter_cents: float  # Estimated pitch scatter between singers
    confidence: float                # Analysis confidence (0-1)


def analyze_spectral(audio: np.ndarray, sr: int) -> SpectralAnalysis:
    """
    Extract spectral characteristics.

    Uses Praat via parselmouth for formant analysis (gold standard).
    Uses librosa for centroid, flux, and rolloff.

    For choral recordings, formants will be smeared due to
    multiple voices - we want the aggregate envelope.

    Args:
        audio: Mono audio signal
        sr: Sample rate in Hz

    Returns:
        SpectralAnalysis with extracted parameters
    """
    import librosa

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Get formants using Praat
    try:
        formant_freqs, formant_bws = _get_formants_praat(audio, sr)
        formant_confidence = 0.9
    except ImportError:
        warnings.warn("parselmouth not available, using librosa fallback for formants")
        formant_freqs, formant_bws = _get_formants_fallback(audio, sr)
        formant_confidence = 0.5
    except Exception as e:
        warnings.warn(f"Formant extraction failed: {e}")
        formant_freqs = [500.0, 1500.0, 2500.0]
        formant_bws = [100.0, 150.0, 200.0]
        formant_confidence = 0.2

    # Spectral features using librosa
    hop_length = int(sr * 0.01)  # 10ms hop
    n_fft = 2048

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)

    # Spectral flux (onset strength as proxy)
    flux = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    flux_mean = np.mean(flux)

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
    )[0]
    rolloff_mean = np.mean(rolloff)

    # Harmonic-to-noise ratio
    try:
        hnr = _calculate_hnr_praat(audio, sr)
        hnr_confidence = 0.9
    except ImportError:
        hnr = _calculate_hnr_fallback(audio, sr)
        hnr_confidence = 0.6
    except Exception:
        hnr = 10.0  # Default typical value
        hnr_confidence = 0.2

    # Chorus width / pitch scatter analysis
    try:
        chorus_metrics = _analyze_chorus_width(audio, sr)
        chorus_confidence = 0.8
    except Exception as e:
        warnings.warn(f"Chorus width analysis failed: {e}")
        chorus_metrics = {
            'peak_width_cents': 15.0,
            'inharmonicity': 0.02,
            'am_depth': 0.1,
            'pitch_scatter_cents': 10.0
        }
        chorus_confidence = 0.2

    # Overall confidence
    confidence = 0.4 * formant_confidence + 0.3 * hnr_confidence + 0.3 * chorus_confidence

    return SpectralAnalysis(
        formant_freqs=formant_freqs,
        formant_bandwidths=formant_bws,
        spectral_centroid_mean=float(centroid_mean),
        spectral_centroid_std=float(centroid_std),
        harmonic_to_noise_ratio=float(hnr),
        spectral_flux_mean=float(flux_mean),
        spectral_rolloff_mean=float(rolloff_mean),
        harmonic_peak_width_cents=float(chorus_metrics['peak_width_cents']),
        inharmonicity=float(chorus_metrics['inharmonicity']),
        amplitude_modulation_depth=float(chorus_metrics['am_depth']),
        estimated_pitch_scatter_cents=float(chorus_metrics['pitch_scatter_cents']),
        confidence=float(confidence)
    )


def _get_formants_praat(audio: np.ndarray, sr: int) -> Tuple[List[float], List[float]]:
    """
    Extract formant frequencies and bandwidths using Praat.

    Praat settings for choral:
    - max_formant: 5500 Hz (assuming mixed choir)
    - num_formants: 5
    - window_length: 0.025s
    """
    import parselmouth
    from parselmouth.praat import call

    # Create Praat Sound object
    snd = parselmouth.Sound(audio, sampling_frequency=sr)

    # Extract formants using Burg method
    # Settings optimized for choral/singing voice
    formant = call(
        snd,
        "To Formant (burg)",
        0.0,    # Time step (0 = auto)
        5,      # Max number of formants
        5500,   # Maximum formant frequency (Hz)
        0.025,  # Window length (s)
        50      # Pre-emphasis from (Hz)
    )

    # Get number of frames
    n_frames = call(formant, "Get number of frames")

    if n_frames == 0:
        raise ValueError("No formant frames extracted")

    # Collect F1, F2, F3 values across all frames
    f1_values, f2_values, f3_values = [], [], []
    b1_values, b2_values, b3_values = [], [], []

    for i in range(1, n_frames + 1):
        t = call(formant, "Get time from frame number", i)

        # Get formant values (0 means undefined)
        f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
        f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
        f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")

        b1 = call(formant, "Get bandwidth at time", 1, t, "Hertz", "Linear")
        b2 = call(formant, "Get bandwidth at time", 2, t, "Hertz", "Linear")
        b3 = call(formant, "Get bandwidth at time", 3, t, "Hertz", "Linear")

        # Filter valid values
        if f1 > 0 and 200 < f1 < 1200:
            f1_values.append(f1)
            if b1 > 0:
                b1_values.append(b1)
        if f2 > 0 and 600 < f2 < 3000:
            f2_values.append(f2)
            if b2 > 0:
                b2_values.append(b2)
        if f3 > 0 and 1500 < f3 < 4500:
            f3_values.append(f3)
            if b3 > 0:
                b3_values.append(b3)

    # Compute medians (robust to outliers)
    f1_median = np.median(f1_values) if f1_values else 500.0
    f2_median = np.median(f2_values) if f2_values else 1500.0
    f3_median = np.median(f3_values) if f3_values else 2500.0

    b1_median = np.median(b1_values) if b1_values else 100.0
    b2_median = np.median(b2_values) if b2_values else 150.0
    b3_median = np.median(b3_values) if b3_values else 200.0

    return (
        [float(f1_median), float(f2_median), float(f3_median)],
        [float(b1_median), float(b2_median), float(b3_median)]
    )


def _get_formants_fallback(audio: np.ndarray, sr: int) -> Tuple[List[float], List[float]]:
    """
    Fallback formant estimation using LPC analysis.

    Less accurate than Praat but works without parselmouth.
    """
    import librosa
    from scipy.signal import lfilter

    # Frame parameters
    frame_length = int(sr * 0.025)  # 25ms
    hop_length = int(sr * 0.01)     # 10ms

    # Pre-emphasis
    pre_emphasis = 0.97
    audio_preemph = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # Frame the signal
    frames = librosa.util.frame(audio_preemph, frame_length=frame_length, hop_length=hop_length)

    formant_estimates = [[], [], []]

    for frame in frames.T:
        # Apply window
        windowed = frame * np.hamming(len(frame))

        # LPC analysis (order = sr/1000 + 2, typical rule)
        lpc_order = min(int(sr / 1000) + 2, 16)

        try:
            # Autocorrelation method for LPC
            autocorr = np.correlate(windowed, windowed, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]

            # Levinson-Durbin
            lpc_coeffs = _levinson_durbin(autocorr, lpc_order)

            # Find roots of LPC polynomial
            roots = np.roots(lpc_coeffs)

            # Keep roots inside unit circle with positive imaginary part
            roots = roots[np.abs(roots) < 1]
            roots = roots[np.imag(roots) > 0]

            # Convert to frequencies
            angles = np.angle(roots)
            freqs = angles * sr / (2 * np.pi)
            freqs = np.sort(freqs)

            # Assign to F1, F2, F3 based on typical ranges
            for f in freqs:
                if 200 < f < 1200:
                    formant_estimates[0].append(f)
                elif 600 < f < 3000:
                    formant_estimates[1].append(f)
                elif 1500 < f < 4500:
                    formant_estimates[2].append(f)
        except:
            continue

    # Compute medians
    f1 = np.median(formant_estimates[0]) if formant_estimates[0] else 500.0
    f2 = np.median(formant_estimates[1]) if formant_estimates[1] else 1500.0
    f3 = np.median(formant_estimates[2]) if formant_estimates[2] else 2500.0

    # Rough bandwidth estimates (typical values)
    return (
        [float(f1), float(f2), float(f3)],
        [100.0, 150.0, 200.0]
    )


def _levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
    """Levinson-Durbin algorithm for LPC coefficients."""
    a = np.zeros(order + 1)
    a[0] = 1.0

    e = r[0]

    for i in range(1, order + 1):
        lambda_val = -np.sum(a[:i] * r[i:0:-1]) / e
        a[1:i+1] = a[1:i+1] + lambda_val * a[i-1::-1]
        e = e * (1 - lambda_val ** 2)

    return a


def _calculate_hnr_praat(audio: np.ndarray, sr: int) -> float:
    """
    Calculate Harmonic-to-Noise Ratio using Praat.

    HNR measures the ratio of harmonic energy to noise energy.
    Higher values indicate cleaner, more periodic signals.
    """
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(audio, sampling_frequency=sr)

    # Create harmonicity object
    harmonicity = call(
        snd,
        "To Harmonicity (cc)",
        0.01,   # Time step (s)
        75,     # Minimum pitch (Hz)
        0.1,    # Silence threshold
        1.0     # Periods per window
    )

    # Get mean HNR (excluding undefined values)
    hnr = call(harmonicity, "Get mean", 0, 0)

    if np.isnan(hnr):
        return 10.0  # Default fallback

    return hnr


def _calculate_hnr_fallback(audio: np.ndarray, sr: int) -> float:
    """
    Fallback HNR estimation using autocorrelation.

    Less accurate than Praat but provides a reasonable estimate.
    """
    import librosa

    # Use harmonic/percussive separation as proxy
    harmonic, percussive = librosa.effects.hpss(audio)

    # Calculate energy ratio
    harmonic_energy = np.sum(harmonic ** 2)
    total_energy = np.sum(audio ** 2)

    if total_energy < 1e-10:
        return 10.0

    # Noise energy approximation
    noise_energy = total_energy - harmonic_energy

    if noise_energy < 1e-10:
        return 30.0  # Very clean

    # HNR in dB
    hnr = 10 * np.log10(harmonic_energy / noise_energy)

    # Clamp to reasonable range
    return np.clip(hnr, 0, 40)


def _analyze_chorus_width(audio: np.ndarray, sr: int) -> dict:
    """
    Analyze chorus width / pitch scatter from harmonic structure.

    Multiple singers on the same note create:
    1. Wider harmonic peaks (frequency smearing)
    2. Inharmonicity (partials deviate from perfect ratios)
    3. Amplitude modulation (beating between slightly detuned voices)

    Returns dict with:
        peak_width_cents: Mean -3dB width of harmonic peaks
        inharmonicity: Deviation from perfect harmonic series (0-1)
        am_depth: Amplitude modulation depth (0-1)
        pitch_scatter_cents: Estimated pitch scatter between singers
    """
    import librosa
    from scipy.signal import find_peaks, peak_widths
    from scipy.ndimage import uniform_filter1d

    # Use longer FFT for better frequency resolution
    n_fft = 8192
    hop_length = n_fft // 4

    # Compute spectrogram
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Average spectrum (more stable than single frames)
    mean_spectrum = np.mean(S, axis=1)
    mean_spectrum_db = librosa.amplitude_to_db(mean_spectrum, ref=np.max)

    # Find fundamental frequency range (focus on vocal range 100-500 Hz)
    vocal_mask = (freqs >= 100) & (freqs <= 500)
    vocal_spectrum = mean_spectrum.copy()
    vocal_spectrum[~vocal_mask] = 0

    # Find the fundamental peak
    f0_idx = np.argmax(vocal_spectrum)
    f0 = freqs[f0_idx]

    if f0 < 80:  # Fallback if detection fails
        f0 = 200  # Reasonable default for mixed choir

    # Analyze harmonic peaks
    peak_widths_cents = []
    inharmonicity_values = []

    # Check first 6 harmonics
    for h in range(1, 7):
        expected_freq = f0 * h
        if expected_freq > sr / 2 - 100:
            break

        # Find the actual peak near expected harmonic
        search_range_hz = expected_freq * 0.1  # ±10% search window
        search_mask = (freqs >= expected_freq - search_range_hz) & \
                      (freqs <= expected_freq + search_range_hz)

        if not np.any(search_mask):
            continue

        search_spectrum = mean_spectrum.copy()
        search_spectrum[~search_mask] = 0
        peak_idx = np.argmax(search_spectrum)
        actual_freq = freqs[peak_idx]

        if actual_freq < 50:
            continue

        # Measure peak width at -3dB
        peak_width_cents = _measure_peak_width_cents(
            mean_spectrum_db, freqs, peak_idx, actual_freq
        )
        if peak_width_cents is not None:
            peak_widths_cents.append(peak_width_cents)

        # Measure inharmonicity (deviation from perfect harmonic)
        if h > 1:
            expected = f0 * h
            deviation_cents = 1200 * np.log2(actual_freq / expected) if expected > 0 else 0
            inharmonicity_values.append(abs(deviation_cents))

    # Calculate amplitude modulation depth from envelope
    am_depth = _calculate_am_depth(audio, sr, f0)

    # Aggregate results
    if peak_widths_cents:
        mean_peak_width = np.median(peak_widths_cents)
    else:
        mean_peak_width = 15.0  # Default

    if inharmonicity_values:
        mean_inharmonicity = np.mean(inharmonicity_values) / 100  # Normalize to 0-1 range
        mean_inharmonicity = min(mean_inharmonicity, 1.0)
    else:
        mean_inharmonicity = 0.02

    # Estimate pitch scatter from peak width
    # Peak width in cents ≈ 2 * pitch_scatter (roughly)
    # Solo voice: ~5-10 cents width
    # Small ensemble: ~15-25 cents
    # Large choir: ~25-50 cents
    estimated_scatter = mean_peak_width / 2

    return {
        'peak_width_cents': mean_peak_width,
        'inharmonicity': mean_inharmonicity,
        'am_depth': am_depth,
        'pitch_scatter_cents': estimated_scatter
    }


def _measure_peak_width_cents(
    spectrum_db: np.ndarray,
    freqs: np.ndarray,
    peak_idx: int,
    peak_freq: float
) -> Optional[float]:
    """
    Measure the -3dB width of a spectral peak in cents.
    """
    peak_level = spectrum_db[peak_idx]
    threshold = peak_level - 3  # -3dB point

    # Search left for -3dB crossing
    left_idx = peak_idx
    while left_idx > 0 and spectrum_db[left_idx] > threshold:
        left_idx -= 1

    # Search right for -3dB crossing
    right_idx = peak_idx
    while right_idx < len(spectrum_db) - 1 and spectrum_db[right_idx] > threshold:
        right_idx += 1

    # Get frequencies at crossings
    left_freq = freqs[left_idx]
    right_freq = freqs[right_idx]

    if left_freq <= 0 or right_freq <= 0 or left_freq >= right_freq:
        return None

    # Convert width to cents
    width_cents = 1200 * np.log2(right_freq / left_freq)

    # Sanity check
    if width_cents < 1 or width_cents > 200:
        return None

    return width_cents


def _calculate_am_depth(audio: np.ndarray, sr: int, f0: float) -> float:
    """
    Calculate amplitude modulation depth from envelope fluctuations.

    Multiple detuned voices create beating patterns visible as AM.
    Higher AM depth suggests more pitch scatter.
    """
    from scipy.signal import hilbert
    from scipy.ndimage import uniform_filter1d

    # Extract envelope using Hilbert transform
    analytic = hilbert(audio)
    envelope = np.abs(analytic)

    # Smooth envelope to remove very fast fluctuations
    smooth_window = int(sr * 0.01)  # 10ms
    if smooth_window > 1:
        envelope = uniform_filter1d(envelope, size=smooth_window)

    # Look for modulation in the 1-20 Hz range (typical beating)
    # Use a longer window for frequency analysis
    window_length = int(sr * 0.5)  # 500ms windows
    hop = window_length // 2

    am_depths = []

    for start in range(0, len(envelope) - window_length, hop):
        segment = envelope[start:start + window_length]

        # Normalize
        if np.max(segment) < 1e-6:
            continue
        segment = segment / np.mean(segment)

        # FFT of envelope
        n_fft = 1024
        env_fft = np.abs(np.fft.rfft(segment - 1, n_fft))  # Remove DC
        env_freqs = np.fft.rfftfreq(n_fft, 1 / sr * (len(segment) / n_fft))

        # Look for energy in modulation band (1-20 Hz)
        mod_mask = (env_freqs >= 1) & (env_freqs <= 20)
        if not np.any(mod_mask):
            continue

        mod_energy = np.sum(env_fft[mod_mask] ** 2)
        total_energy = np.sum(env_fft ** 2) + 1e-10

        am_depth = np.sqrt(mod_energy / total_energy)
        am_depths.append(am_depth)

    if am_depths:
        return float(np.median(am_depths))
    else:
        return 0.1  # Default
