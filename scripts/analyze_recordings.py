#!/usr/bin/env python3
"""
Analyze choral recordings to extract acoustic parameters
for calibrating synthetic data augmentation.

This tool extracts:
- Reverb characteristics (RT60, wet/dry ratio)
- Pitch analysis (vibrato rate/depth, pitch scatter, drift)
- Spectral profile (formants, HNR, centroid)
- Dynamic range (crest factor, compression artifacts)

Usage:
    python scripts/analyze_recordings.py data/recordings/*.wav
    python scripts/analyze_recordings.py data/recordings/*.mp3
    python scripts/analyze_recordings.py -i data/recordings -o data/analysis

Example workflow:
    1. Place choral recordings in data/recordings/
    2. Run: python scripts/analyze_recordings.py data/recordings/*.wav
    3. Review individual reports in data/analysis/
    4. Use summary.json to calibrate augmentation parameters
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.reverb import analyze_reverb
from src.analysis.pitch import analyze_pitch
from src.analysis.spectral import analyze_spectral
from src.analysis.dynamics import analyze_dynamics
from src.analysis.report import (
    RecordingAnalysis,
    generate_report,
    summarize_analyses,
    generate_augmentation_config,
    print_summary,
)


def analyze_file(filepath: Path, verbose: bool = True) -> RecordingAnalysis:
    """
    Run full analysis on a single recording.

    Args:
        filepath: Path to audio file
        verbose: Print progress messages

    Returns:
        RecordingAnalysis with all extracted parameters
    """
    if verbose:
        print(f"\nAnalyzing: {filepath.name}")

    # Load audio (mono, preserve original sample rate)
    audio, sr = librosa.load(filepath, sr=None, mono=True)
    duration = len(audio) / sr

    if verbose:
        print(f"  Duration: {duration:.1f}s, Sample rate: {sr} Hz")

    # Run analyses
    if verbose:
        print("  Analyzing reverb...")
    reverb = analyze_reverb(audio, sr)

    if verbose:
        print("  Analyzing pitch...")
    pitch = analyze_pitch(audio, sr)

    if verbose:
        print("  Analyzing spectral...")
    spectral = analyze_spectral(audio, sr)

    if verbose:
        print("  Analyzing dynamics...")
    dynamics = analyze_dynamics(audio, sr)

    return RecordingAnalysis(
        filename=filepath.name,
        duration_sec=duration,
        sample_rate=sr,
        reverb=reverb,
        pitch=pitch,
        spectral=spectral,
        dynamics=dynamics,
        metadata={
            "filepath": str(filepath.absolute()),
            "channels_original": "mono (converted)" if True else "stereo",
        }
    )


def find_audio_files(input_path: Path) -> list[Path]:
    """Find all audio files in directory or return single file."""
    supported_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        files = []
        for ext in supported_extensions:
            files.extend(input_path.glob(f'*{ext}'))
            files.extend(input_path.glob(f'*{ext.upper()}'))
        return sorted(files)
    else:
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Analyze choral recordings for acoustic parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s data/recordings/*.wav
    %(prog)s -i data/recordings -o data/analysis
    %(prog)s recording.mp3 --no-summary

Obtaining Test Recordings:
    Option 1: YouTube download (for personal research use)
        pip install yt-dlp
        yt-dlp -x --audio-format wav -o "data/recordings/%%(title)s.%%(ext)s" <URL>

    Option 2: CD rips from your collection

    Suggested recordings for analysis:
    - Lauridsen - O Magnum Mysterium (LA Master Chorale or Polyphony)
    - Tallis - Spem in Alium (Tallis Scholars)
    - Whitacre - Lux Aurumque (BYU Singers)
    - Bach - Mass in B minor, Kyrie (Monteverdi Choir / Gardiner)
        """
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Audio files to analyze (supports glob patterns)"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        help="Input directory containing audio files"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("data/analysis"),
        help="Output directory for reports (default: data/analysis)"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip summary generation"
    )
    parser.add_argument(
        "--augmentation-config",
        action="store_true",
        help="Generate augmentation configuration file"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Collect input files
    input_files = []

    if args.input:
        input_files.extend(find_audio_files(args.input))

    for file_pattern in args.files:
        path = Path(file_pattern)
        if path.exists():
            input_files.extend(find_audio_files(path))
        else:
            # Try glob pattern
            parent = path.parent if path.parent.exists() else Path(".")
            pattern = path.name
            input_files.extend(parent.glob(pattern))

    # Remove duplicates while preserving order
    seen = set()
    input_files = [f for f in input_files if not (f in seen or seen.add(f))]

    if not input_files:
        print("Error: No audio files found.")
        print("Provide files as arguments or use -i to specify input directory.")
        sys.exit(1)

    print(f"Found {len(input_files)} audio file(s) to analyze")

    # Create output directory
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze each file
    analyses = []

    for filepath in tqdm(input_files, desc="Analyzing recordings", disable=args.quiet):
        filepath = Path(filepath)

        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping")
            continue

        try:
            analysis = analyze_file(filepath, verbose=not args.quiet)
            analyses.append(analysis)

            # Save individual report
            report_path = output_dir / f"{filepath.stem}_analysis.json"
            generate_report(analysis, report_path)

            if not args.quiet:
                print(f"  Saved: {report_path}")

        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            continue

    # Generate summary
    if analyses and not args.no_summary:
        summary = summarize_analyses(analyses)

        # Save summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")

        # Print summary
        if not args.quiet:
            print_summary(summary)

        # Generate augmentation config if requested
        if args.augmentation_config:
            aug_config = generate_augmentation_config(summary)
            aug_config_path = output_dir / "augmentation_config.json"
            with open(aug_config_path, 'w') as f:
                json.dump(aug_config, f, indent=2)
            print(f"Augmentation config saved to: {aug_config_path}")

        # Print key findings for quick reference
        if not args.quiet:
            print("\n" + "=" * 60)
            print("KEY FINDINGS FOR AUGMENTATION CALIBRATION")
            print("=" * 60)
            if "rt60" in summary:
                print(f"RT60 range: {summary['rt60']['min']:.1f}s - {summary['rt60']['max']:.1f}s")
            if "vibrato_rate" in summary:
                print(f"Vibrato rate: {summary['vibrato_rate']['mean']:.1f} Hz")
            if "vibrato_depth" in summary:
                print(f"Vibrato depth: {summary['vibrato_depth']['mean']:.0f} cents")
            if "pitch_stability" in summary:
                print(f"Pitch scatter: ±{summary['pitch_stability']['mean']:.0f} cents")
            if "crest_factor" in summary:
                print(f"Crest factor: {summary['crest_factor']['mean']:.1f} dB")

    print(f"\nAnalyzed {len(analyses)} recording(s)")


if __name__ == "__main__":
    main()
