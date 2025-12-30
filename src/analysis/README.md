# Acoustic Analysis Module

Extracts acoustic characteristics from choral recordings to calibrate augmentation parameters for synthetic data generation.

## Features

### Reverb Analysis (`reverb.py`)
- **RT60 estimation**: Reverberation time using Schroeder backward integration
- **Early Decay Time (EDT)**: Initial decay characteristics
- **Clarity C50**: Ratio of early to late energy
- **Wet/dry ratio**: Estimated reverb mix level

### Pitch Analysis (`pitch.py`)
- **CREPE neural pitch tracking**: Requires Python 3.12 + TensorFlow
- **Vibrato extraction**: Rate (Hz) and depth (cents)
- **Pitch stability**: Standard deviation over time
- **Pitch drift**: Long-term pitch tendency

### Spectral Analysis (`spectral.py`)
- **Formant frequencies**: F1, F2, F3 via Praat/parselmouth
- **Harmonic-to-noise ratio (HNR)**: Voice quality metric
- **Spectral centroid/flux**: Timbral characteristics
- **Chorus width estimation**:
  - Harmonic peak width (cents): Spread of harmonic peaks due to pitch variation
  - Inharmonicity: Deviation from perfect harmonic series (0-1)
  - Amplitude modulation depth: Beating from pitch differences
  - Estimated pitch scatter: Derived from harmonic analysis

### Dynamics Analysis (`dynamics.py`)
- **Peak/RMS levels**: In dB
- **Crest factor**: Peak-to-RMS ratio
- **LUFS integrated loudness**: ITU-R BS.1770 standard
- **Dynamic range**: Loudness variation
- **Compression estimate**: None/light/moderate/heavy

## Usage

```bash
# Analyze recordings in data/recordings/
python scripts/analyze_recordings.py

# Output written to data/analysis/
# - Individual JSON files per recording
# - summary.json with aggregate statistics
# - augmentation_config.json with calibrated parameter ranges
```

## Requirements

- Python 3.12 (required for TensorFlow/CREPE compatibility)
- See `requirements.txt` for dependencies

## Output Format

Each recording produces a JSON file with structure:
```json
{
  "filename": "recording.wav",
  "duration_sec": 260.6,
  "reverb": {"rt60_estimate": 2.5, "wet_dry_estimate": 0.55, ...},
  "pitch": {"vibrato_rate_hz": 5.5, "vibrato_depth_cents": 25, ...},
  "spectral": {
    "formant_freqs": [500, 1400, 2700],
    "harmonic_peak_width_cents": 65,
    "estimated_pitch_scatter_cents": 33,
    ...
  },
  "dynamics": {"crest_factor_db": 19, "lufs_integrated": -26, ...}
}
```

## Calibrated Ranges

The `augmentation_config.json` provides parameter ranges for data augmentation:

| Parameter | Typical Range | Unit |
|-----------|--------------|------|
| RT60 | 1.3 - 3.9 | seconds |
| Wet/dry | 0.4 - 0.7 | ratio |
| Vibrato rate | 4.5 - 7.0 | Hz |
| Vibrato depth | 20 - 65 | cents |
| Pitch scatter | 16 - 50 | cents std |
| Crest factor | 15 - 24 | dB |
| LUFS | -33 to -20 | LUFS |
