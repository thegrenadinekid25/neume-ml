"""
Report generation for acoustic analysis.

Aggregates results from individual analyses and generates JSON reports.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import numpy as np

from .reverb import ReverbAnalysis
from .pitch import PitchAnalysis
from .spectral import SpectralAnalysis
from .dynamics import DynamicsAnalysis


@dataclass
class RecordingAnalysis:
    """Complete analysis results for a single recording."""
    filename: str
    duration_sec: float
    sample_rate: int
    reverb: ReverbAnalysis
    pitch: PitchAnalysis
    spectral: SpectralAnalysis
    dynamics: DynamicsAnalysis
    metadata: Dict[str, Any] = field(default_factory=dict)


def generate_report(analysis: RecordingAnalysis, output_path: Path) -> None:
    """
    Save analysis as JSON.

    Args:
        analysis: RecordingAnalysis to save
        output_path: Path for output JSON file
    """
    # Convert dataclass to dict
    data = _dataclass_to_dict(analysis)

    # Write with nice formatting
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def _dataclass_to_dict(obj: Any) -> Any:
    """
    Recursively convert dataclasses to dicts.

    Handles nested dataclasses and numpy types.
    """
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def summarize_analyses(analyses: List[RecordingAnalysis]) -> Dict[str, Any]:
    """
    Generate summary statistics across all recordings.

    Output ranges and means for each parameter that can be
    used to calibrate augmentation parameters.

    Args:
        analyses: List of RecordingAnalysis objects

    Returns:
        Dictionary with summary statistics
    """
    if not analyses:
        return {}

    summary = {
        "n_recordings": len(analyses),
        "total_duration_sec": sum(a.duration_sec for a in analyses),
    }

    # Collect values for each metric
    metrics = {
        # Reverb metrics
        "rt60": [a.reverb.rt60_estimate for a in analyses],
        "early_decay_time": [a.reverb.early_decay_time for a in analyses],
        "clarity_c50": [a.reverb.clarity_c50 for a in analyses],
        "wet_dry": [a.reverb.wet_dry_estimate for a in analyses],

        # Pitch metrics
        "vibrato_rate": [a.pitch.vibrato_rate_hz for a in analyses],
        "vibrato_depth": [a.pitch.vibrato_depth_cents for a in analyses],
        "pitch_stability": [a.pitch.pitch_stability_std_cents for a in analyses],
        "pitch_drift": [a.pitch.drift_cents_per_second for a in analyses],

        # Spectral metrics
        "spectral_centroid": [a.spectral.spectral_centroid_mean for a in analyses],
        "hnr": [a.spectral.harmonic_to_noise_ratio for a in analyses],
        "spectral_flux": [a.spectral.spectral_flux_mean for a in analyses],
        "f1": [a.spectral.formant_freqs[0] for a in analyses if a.spectral.formant_freqs],
        "f2": [a.spectral.formant_freqs[1] for a in analyses if len(a.spectral.formant_freqs) > 1],
        "f3": [a.spectral.formant_freqs[2] for a in analyses if len(a.spectral.formant_freqs) > 2],

        # Chorus width / pitch scatter metrics
        "harmonic_peak_width_cents": [a.spectral.harmonic_peak_width_cents for a in analyses],
        "inharmonicity": [a.spectral.inharmonicity for a in analyses],
        "am_depth": [a.spectral.amplitude_modulation_depth for a in analyses],
        "estimated_pitch_scatter_cents": [a.spectral.estimated_pitch_scatter_cents for a in analyses],

        # Dynamics metrics
        "peak_db": [a.dynamics.peak_db for a in analyses],
        "rms_db": [a.dynamics.rms_db for a in analyses],
        "crest_factor": [a.dynamics.crest_factor_db for a in analyses],
        "dynamic_range": [a.dynamics.dynamic_range_db for a in analyses],
        "lufs": [a.dynamics.lufs_integrated for a in analyses],
    }

    # Calculate statistics for each metric
    for metric_name, values in metrics.items():
        if values:
            values_array = np.array(values)
            summary[metric_name] = {
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "median": float(np.median(values_array)),
            }

    # Compression classification distribution
    compression_counts = {}
    for a in analyses:
        c = a.dynamics.compression_estimate
        compression_counts[c] = compression_counts.get(c, 0) + 1
    summary["compression_distribution"] = compression_counts

    # Confidence statistics
    summary["confidence"] = {
        "reverb": {
            "mean": float(np.mean([a.reverb.confidence for a in analyses])),
            "min": float(np.min([a.reverb.confidence for a in analyses])),
        },
        "pitch": {
            "mean": float(np.mean([a.pitch.confidence for a in analyses])),
            "min": float(np.min([a.pitch.confidence for a in analyses])),
        },
        "spectral": {
            "mean": float(np.mean([a.spectral.confidence for a in analyses])),
            "min": float(np.min([a.spectral.confidence for a in analyses])),
        },
    }

    return summary


def generate_augmentation_config(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate recommended augmentation parameters from summary statistics.

    These parameters can be used directly in the augmentation pipeline.

    Args:
        summary: Summary statistics from summarize_analyses()

    Returns:
        Dictionary with augmentation parameter recommendations
    """
    config = {
        "reverb": {},
        "pitch": {},
        "dynamics": {},
        "description": "Augmentation parameters derived from acoustic analysis of target recordings."
    }

    # Reverb parameters
    if "rt60" in summary:
        rt60 = summary["rt60"]
        # Add some margin around observed range
        config["reverb"] = {
            "rt60_range_sec": [
                max(0.3, rt60["min"] * 0.8),
                min(10.0, rt60["max"] * 1.2)
            ],
            "wet_dry_range": [
                max(0.0, summary.get("wet_dry", {}).get("min", 0.1) - 0.1),
                min(1.0, summary.get("wet_dry", {}).get("max", 0.5) + 0.1)
            ],
        }

    # Pitch parameters
    if "vibrato_rate" in summary and "vibrato_depth" in summary:
        # Use estimated_pitch_scatter from harmonic analysis (more reliable than pitch tracker)
        scatter_stats = summary.get("estimated_pitch_scatter_cents", {})
        scatter_min = scatter_stats.get("min", 8)
        scatter_max = scatter_stats.get("max", 25)

        config["pitch"] = {
            "vibrato_rate_range_hz": [
                max(4.0, summary["vibrato_rate"]["min"] - 0.5),
                min(9.0, summary["vibrato_rate"]["max"] + 0.5)
            ],
            "vibrato_depth_range_cents": [
                max(20, summary["vibrato_depth"]["min"] - 15),
                min(150, summary["vibrato_depth"]["max"] + 15)
            ],
            # Pitch scatter from harmonic peak width analysis
            "pitch_scatter_std_cents": [
                max(5, scatter_min - 3),
                min(50, scatter_max + 5)
            ],
        }

        # Also include chorus width metrics for reference
        if "harmonic_peak_width_cents" in summary:
            config["pitch"]["harmonic_peak_width_cents"] = [
                summary["harmonic_peak_width_cents"]["min"],
                summary["harmonic_peak_width_cents"]["max"]
            ]
        if "am_depth" in summary:
            config["pitch"]["am_depth_range"] = [
                summary["am_depth"]["min"],
                summary["am_depth"]["max"]
            ]

    # Dynamics parameters
    if "crest_factor" in summary:
        config["dynamics"] = {
            "target_crest_factor_range_db": [
                summary["crest_factor"]["min"] - 2,
                summary["crest_factor"]["max"] + 2
            ],
            "target_lufs_range": [
                summary.get("lufs", {}).get("min", -24) - 3,
                summary.get("lufs", {}).get("max", -14) + 3
            ],
        }

    return config


def print_summary(summary: Dict[str, Any]) -> None:
    """
    Print a formatted summary of the analysis results.

    Args:
        summary: Summary statistics from summarize_analyses()
    """
    print("\n" + "=" * 60)
    print("ACOUSTIC ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nRecordings analyzed: {summary.get('n_recordings', 0)}")
    print(f"Total duration: {summary.get('total_duration_sec', 0):.1f}s")

    print("\n--- REVERB CHARACTERISTICS ---")
    if "rt60" in summary:
        rt60 = summary["rt60"]
        print(f"RT60: {rt60['min']:.1f}s - {rt60['max']:.1f}s (mean: {rt60['mean']:.1f}s)")
    if "wet_dry" in summary:
        wd = summary["wet_dry"]
        print(f"Wet/Dry ratio: {wd['min']:.2f} - {wd['max']:.2f} (mean: {wd['mean']:.2f})")

    print("\n--- PITCH CHARACTERISTICS ---")
    if "vibrato_rate" in summary:
        vr = summary["vibrato_rate"]
        print(f"Vibrato rate: {vr['min']:.1f} - {vr['max']:.1f} Hz (mean: {vr['mean']:.1f} Hz)")
    if "vibrato_depth" in summary:
        vd = summary["vibrato_depth"]
        print(f"Vibrato depth: {vd['min']:.0f} - {vd['max']:.0f} cents (mean: {vd['mean']:.0f} cents)")
    if "pitch_stability" in summary:
        ps = summary["pitch_stability"]
        print(f"Pitch scatter: ±{ps['min']:.0f} - ±{ps['max']:.0f} cents (mean: ±{ps['mean']:.0f} cents)")

    print("\n--- SPECTRAL CHARACTERISTICS ---")
    if "spectral_centroid" in summary:
        sc = summary["spectral_centroid"]
        print(f"Spectral centroid: {sc['min']:.0f} - {sc['max']:.0f} Hz (mean: {sc['mean']:.0f} Hz)")
    if "hnr" in summary:
        hnr = summary["hnr"]
        print(f"Harmonic-to-noise ratio: {hnr['min']:.1f} - {hnr['max']:.1f} dB (mean: {hnr['mean']:.1f} dB)")
    if "f1" in summary and "f2" in summary:
        print(f"Formants: F1={summary['f1']['mean']:.0f}Hz, F2={summary['f2']['mean']:.0f}Hz, F3={summary.get('f3', {}).get('mean', 2500):.0f}Hz")

    print("\n--- CHORUS WIDTH / PITCH SCATTER ---")
    if "harmonic_peak_width_cents" in summary:
        pw = summary["harmonic_peak_width_cents"]
        print(f"Harmonic peak width: {pw['min']:.1f} - {pw['max']:.1f} cents (mean: {pw['mean']:.1f} cents)")
    if "estimated_pitch_scatter_cents" in summary:
        ps = summary["estimated_pitch_scatter_cents"]
        print(f"Estimated pitch scatter: ±{ps['min']:.1f} - ±{ps['max']:.1f} cents (mean: ±{ps['mean']:.1f} cents)")
    if "am_depth" in summary:
        am = summary["am_depth"]
        print(f"AM depth (beating): {am['min']:.3f} - {am['max']:.3f} (mean: {am['mean']:.3f})")
    if "inharmonicity" in summary:
        ih = summary["inharmonicity"]
        print(f"Inharmonicity: {ih['min']:.3f} - {ih['max']:.3f} (mean: {ih['mean']:.3f})")

    print("\n--- DYNAMICS CHARACTERISTICS ---")
    if "crest_factor" in summary:
        cf = summary["crest_factor"]
        print(f"Crest factor: {cf['min']:.1f} - {cf['max']:.1f} dB (mean: {cf['mean']:.1f} dB)")
    if "dynamic_range" in summary:
        dr = summary["dynamic_range"]
        print(f"Dynamic range: {dr['min']:.1f} - {dr['max']:.1f} dB (mean: {dr['mean']:.1f} dB)")
    if "lufs" in summary:
        lufs = summary["lufs"]
        print(f"Integrated loudness: {lufs['min']:.1f} - {lufs['max']:.1f} LUFS (mean: {lufs['mean']:.1f} LUFS)")
    if "compression_distribution" in summary:
        dist = summary["compression_distribution"]
        dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(dist.items()))
        print(f"Compression: {dist_str}")

    print("\n--- ANALYSIS CONFIDENCE ---")
    if "confidence" in summary:
        conf = summary["confidence"]
        print(f"Reverb: {conf['reverb']['mean']:.2f} (min: {conf['reverb']['min']:.2f})")
        print(f"Pitch: {conf['pitch']['mean']:.2f} (min: {conf['pitch']['min']:.2f})")
        print(f"Spectral: {conf['spectral']['mean']:.2f} (min: {conf['spectral']['min']:.2f})")

    print("\n" + "=" * 60)
