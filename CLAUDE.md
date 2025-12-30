# Neume-ML - Project Context for Claude Code

## Overview
Neume-ML generates synthetic choral chord audio data for training ML chord recognition models. Template-based approaches fail on choral recordings due to vibrato, reverb, and ensemble effects—this project creates realistic training data using FluidSynth SATB rendering.

## Tech Stack

**Core:**
- Python 3.12 (required for TensorFlow/CREPE compatibility)
- FluidSynth (via pyfluidsynth) for audio synthesis
- NumPy/SciPy for audio processing
- PyYAML for configuration

**Audio I/O:**
- soundfile for WAV read/write

**Acoustic Analysis:**
- librosa for audio features
- CREPE (TensorFlow) for neural pitch tracking
- praat-parselmouth for formant analysis

## Local Development

```bash
# Setup virtual environment (use Python 3.12 specifically)
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install FluidSynth (macOS)
brew install fluid-synth

# Download soundfont
# Get FluidR3_GM.sf2 from https://member.keymusician.com/Member/FluidR3_GM/index.html
# Place in data/soundfonts/FluidR3_GM.sf2

# Generate test samples
python scripts/generate_test_samples.py
```

## Key Directories

```
neume-ml/
├── src/
│   ├── synthesis/
│   │   ├── chord_types.py      # ChordSpec, ChordQuality, pitch classes
│   │   ├── voicing.py          # SATBVoicing, voice leading rules
│   │   └── fluidsynth_renderer.py  # Audio rendering engine
│   ├── augmentation/           # Audio augmentation pipeline
│   │   ├── pipeline.py         # Main orchestrator, AugmentationConfig
│   │   ├── pitch_scatter.py    # Per-voice pitch detuning
│   │   ├── vibrato.py          # LFO pitch modulation
│   │   ├── amplitude_modulation.py  # Beating/tremolo effects
│   │   ├── reverb.py           # Algorithmic + convolution reverb
│   │   ├── dynamics.py         # LUFS normalization, crest factor
│   │   └── lowpass.py          # Variable brightness filter
│   └── utils/
│       └── audio.py            # Config loading, path utilities
├── data/
│   ├── soundfonts/             # .sf2 files (gitignored)
│   ├── impulse_responses/      # IR files for convolution reverb
│   └── output/                 # Generated samples (gitignored)
├── scripts/
│   ├── generate_test_samples.py
│   └── test_augmentation.py    # A/B comparison tests
└── configs/
    └── synthesis.yaml          # Voice ranges, sample rate, soundfont config
```

## Chord Vocabulary

**Phase 1:** 36 chord types
- 12 roots: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
- 3 qualities: major, minor, dominant 7th

**Future phases:** diminished, augmented, maj7, min7, sus2, sus4, extensions

## Voice Ranges (MIDI)

| Voice    | Low  | High | Notes        |
|----------|------|------|--------------|
| Bass     | 40   | 60   | E2 to C4     |
| Tenor    | 48   | 67   | C3 to G4     |
| Alto     | 55   | 74   | G3 to D5     |
| Soprano  | 60   | 79   | C4 to G5     |

## Common Tasks

**Generate test samples:**
```bash
python scripts/generate_test_samples.py
```

**Add a new chord quality:**
1. Add to `ChordQuality` enum in `chord_types.py`
2. Add interval pattern to `QUALITY_INTERVALS` dict
3. Voicing and rendering work automatically

## Big Changes Workflow

For significant features or refactors, use this Claude Code workflow:

1. **Plan with Opus** - Enter plan mode and use an Opus subagent to:
   - Explore the codebase thoroughly
   - Design the implementation approach
   - Identify all files to create/modify/delete
   - Consider edge cases and testing strategy

2. **Review the plan** - Get user approval before implementing

3. **Execute with Haiku** - Use Haiku subagents for grunt work:
   - File parsing and exploration
   - Repetitive code changes
   - Running tests and builds
   - Cleanup tasks

This maximizes quality (Opus planning) while minimizing cost (Haiku execution).

## Augmentation Pipeline

Transforms clean FluidSynth output into realistic choral recordings. Parameters calibrated from professional recordings (Lauridsen, Tallis, Whitacre, Bach).

**Effects chain:**
1. **Pitch scatter** (16-50 cents) - Per-voice detuning for ensemble sound
2. **Vibrato** (4.6-7 Hz, 20-65 cents) - LFO pitch modulation with delayed onset
3. **Amplitude modulation** (39-48% depth) - Beating from phase interference
4. **Low-pass filter** (5-20 kHz) - Variable brightness control
5. **Reverb** (RT60 1.3-3.9s, 39-71% wet) - Algorithmic or convolution
6. **Dynamics** (LUFS -33 to -20) - Loudness normalization

**Usage:**
```python
from src.augmentation import AugmentationPipeline, AugmentationConfig

# Random config within calibrated ranges
config = AugmentationConfig.random(seed=42)

# Or use presets: "minimal", "moderate", "heavy", "cathedral"
config = AugmentationConfig.from_preset("cathedral")

pipeline = AugmentationPipeline(config)
augmented = pipeline.process_mixed(audio, sample_rate)
```

**Test augmentation:**
```bash
python scripts/test_augmentation.py
# Listen to files in data/test_augmentation/
```

## Notes

- Voicing is deterministic given a seed (for reproducibility)
- FluidSynth uses 4 MIDI channels (one per voice) to avoid voice stealing
- Choir Aahs (program 52) is default; Voice Oohs (53) also available
- Audio normalized to -1 to 1 float32
- Augmentation pipeline is reproducible with seed parameter
