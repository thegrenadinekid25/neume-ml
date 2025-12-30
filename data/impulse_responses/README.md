# Impulse Responses for Convolution Reverb

This directory contains impulse response (IR) files for convolution reverb processing.
The augmentation pipeline will automatically load all `.wav` files from this directory.

## Free Sources

### 1. OpenAIR (openairlib.net)
High-quality academic impulse responses from various real spaces.
- License: CC-BY (attribution required)
- Includes cathedrals, concert halls, and churches
- Well-documented acoustic measurements

### 2. EchoThief (echothief.com)
Free IR collection covering many space types.
- License: Free for non-commercial use
- Variety of natural and artificial spaces

### 3. Fokke van Saane (fokkie.home.xs4all.nl)
Free church and cathedral IRs.
- Excellent for choral music applications
- Multiple microphone positions available

### 4. Voxengo (voxengo.com/impulses/)
Free impulse response collection.
- License: Free for commercial and non-commercial use
- Variety of reverb types

## Recommended IRs for Choral Music

When selecting impulse responses for choral augmentation, prioritize:

1. **Large Churches & Cathedrals** (RT60: 2-5s)
   - St. George's Church
   - York Minster
   - Lady Chapel, Ely Cathedral
   - Notre Dame (various)

2. **Concert Halls** (RT60: 1.5-2.5s)
   - Boston Symphony Hall
   - Konzerthaus Berlin
   - Concertgebouw Amsterdam

3. **Chapel/Medium Spaces** (RT60: 1-2s)
   - King's College Chapel
   - Trinity College Chapel

4. **Extreme Spaces** (for variety)
   - Hamilton Mausoleum (15s RT60)
   - Underground cisterns

## Format Requirements

The augmentation pipeline requires:
- **Format**: WAV (uncompressed PCM)
- **Sample Rate**: 44.1kHz or 48kHz preferred
- **Channels**: Mono recommended (stereo will be converted to mono)
- **Bit Depth**: 16-bit or higher

## Directory Structure

Place IR files directly in this directory:

```
data/impulse_responses/
├── README.md (this file)
├── cathedral_large.wav
├── church_medium.wav
├── concert_hall.wav
└── ...
```

## Usage

The pipeline will automatically use these IRs when configured:

```python
from src.augmentation import AugmentationConfig, AugmentationPipeline

config = AugmentationConfig(
    use_convolution_reverb=True,
    ir_directory="data/impulse_responses",
    reverb_wet_dry=0.55,
)

pipeline = AugmentationPipeline(config)
```

If no IRs are available, the pipeline falls back to algorithmic reverb.

## Creating Your Own IRs

To record your own impulse responses:

1. Use a starter pistol, balloon pop, or sine sweep
2. Record in the space with a quality microphone
3. For sine sweep method, deconvolve to get IR
4. Trim silence from beginning
5. Normalize peak to -1dB
6. Save as mono WAV

Tools for IR capture:
- Altiverb IR Pre-processor
- Voxengo Deconvolver
- Room EQ Wizard (REW)
