# Neume-ML

Synthetic choral chord data generation for training ML chord recognition models.

Template-based chord recognition fails on choral recordings due to vibrato, reverb, and ensemble effects. Neume-ML generates realistic training data using FluidSynth SATB/N-voice rendering with comprehensive voicing controls.

## Features

- **32 chord types**: triads, 7ths, extensions, altered, quartal, clusters
- **4-16 voice textures**: from SATB to large ensemble
- **5 bass note types**: root position, inversions, slash chords
- **5 shell voicing strategies**: classical, jazz, rootless A/B, quartal
- **4 voicing styles**: closed, open, mixed, wide
- **5 non-chord tone types**: passing, neighbor, suspension, anticipation, appoggiatura
- **Weighted sampling**: realistic distribution of musical parameters

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/neume-ml.git
cd neume-ml
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install FluidSynth (macOS)
brew install fluid-synth

# Download soundfont (141 MB)
# Get FluidR3_GM.sf2 from https://member.keymusician.com/Member/FluidR3_GM/index.html
mkdir -p data/soundfonts
# Place FluidR3_GM.sf2 in data/soundfonts/

# Run tests
python scripts/test_voicing_system.py

# Generate training data
python src/data_generation/generate_training_set.py --num-samples 1000
```

## Project Structure

```
neume-ml/
├── src/
│   ├── data_generation/
│   │   ├── voicing/
│   │   │   ├── chord_types.py       # 32 chord type definitions
│   │   │   ├── bass_note.py         # Bass note selection & inversions
│   │   │   ├── shell_strategies.py  # 5 shell voicing strategies
│   │   │   ├── duration.py          # 5 duration categories (0.5-5s)
│   │   │   ├── voicing_styles.py    # Open/closed/mixed/wide
│   │   │   ├── non_chord_tones.py   # NCT types and densities
│   │   │   ├── voice_distribution.py # Main voicing algorithm
│   │   │   ├── validation.py        # Sample constraint validation
│   │   │   └── weights.py           # Distribution weights
│   │   └── generate_training_set.py # Main generation script
│   ├── synthesis/
│   │   ├── chord_types.py           # ChordSpec, ChordQuality
│   │   ├── voicing.py               # SATB/NVoice voicing classes
│   │   └── fluidsynth_renderer.py   # Audio rendering engine
│   ├── analysis/                    # Audio analysis tools
│   └── utils/
│       └── audio.py                 # Config & path utilities
├── scripts/
│   ├── test_voicing_system.py       # Comprehensive test suite
│   ├── generate_test_samples.py     # Basic sample generation
│   └── generate_nvoice_samples.py   # N-voice sample generation
├── configs/
│   └── synthesis.yaml               # Voice ranges, sample rate config
└── data/
    ├── soundfonts/                  # .sf2 files (gitignored)
    └── output/                      # Generated samples (gitignored)
```

## Chord Types (32 total)

| Category | Types |
|----------|-------|
| Triads | major, minor, diminished, augmented |
| 7ths | maj7, dom7, min7, minmaj7, half_dim7, dim7 |
| Suspended | sus2, sus4 |
| Added | add9, add11, madd9, 6, m6 |
| Extended | maj9, dom9, min9, dom11, min11, dom13, maj13 |
| Altered | dom7b9, dom7#9, dom7b5, dom7#5 |
| Special | quartal3, quartal4, cluster_whole, cluster_chromatic |

## Voice Distribution

The voicing system distributes pitch classes across N voices (4-16) with:

- **Ascending order**: voices always go from low to high
- **No unisons**: each MIDI note is unique
- **Minimum spacing**: 2 semitones (1 for >10 voices)
- **Range limits**: MIDI 36-88 (C2 to E6)
- **Register selection**: low, medium, high, extended

## API Usage

```python
from data_generation.voicing import (
    CHORD_TYPES,
    get_pitch_classes,
    select_bass_note,
    distribute_voices,
    validate_sample,
)

# Get pitch classes for C major 7
pitch_classes = get_pitch_classes("maj7", root=0)  # [0, 4, 7, 11]

# Distribute across 8 voices
voices = distribute_voices(
    pitch_classes=pitch_classes,
    bass_note=0,  # C in bass
    num_parts=8,
    register="medium",
)

# Validate the voicing
errors = validate_sample(voices, {"num_parts": 8, "expected_pitch_classes": pitch_classes})
assert len(errors) == 0
```

## Shell Voicing Strategies

| Strategy | Priority | Use Case |
|----------|----------|----------|
| classical | root, 5th, 3rd, 7th | Traditional harmony |
| jazz_shell | 3rd, 7th, root | Compact jazz voicings |
| rootless_a | 3rd, 5th, 7th, 9th | Modern jazz (Bill Evans) |
| rootless_b | 7th, 9th, 3rd, 5th | Alternative rootless |
| quartal | stacked 4ths | Modal/contemporary |

## Non-Chord Tones

- **passing**: stepwise motion between chord tones
- **neighbor**: step away and return
- **suspension**: held from previous, resolves down
- **anticipation**: arrives early
- **appoggiatura**: leap to dissonance, resolve by step

Density levels: none, sparse (10%), moderate (25%), dense (40%)

## Configuration

Edit `configs/synthesis.yaml`:

```yaml
synthesis:
  sample_rate: 44100
  soundfont:
    path: data/soundfonts/FluidR3_GM.sf2
  voices:
    bass: { low: 40, high: 60 }
    tenor: { low: 48, high: 67 }
    alto: { low: 55, high: 74 }
    soprano: { low: 60, high: 79 }
```

## Requirements

- Python 3.12+
- FluidSynth (system library)
- pyfluidsynth
- numpy
- soundfile
- pyyaml

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `python scripts/test_voicing_system.py`
4. Submit a pull request
