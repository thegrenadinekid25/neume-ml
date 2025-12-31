"""Conformer-based chord recognition model."""

from .chord_model import ChordModelConfig, ChordRecognitionModel, create_model
from .configs import LARGE_CONFIG, MEDIUM_CONFIG, SMALL_CONFIG, get_config
from .conformer import ConformerBlock
from .heads import ChordRecognitionHeads, ClassificationHead
from .labels import (
    CHORD_TYPE_TO_IDX,
    IDX_TO_CHORD_TYPE,
    IDX_TO_PITCH_CLASS,
    PITCH_CLASS_TO_IDX,
    format_chord_name,
)
from .loss import ChordRecognitionLoss, FocalLoss
from .preprocessing import AudioToFeatures, CQTNormalization, CQTProcessor
from .subsampling import ConvSubsampling

__all__ = [
    "AudioToFeatures",
    "CHORD_TYPE_TO_IDX",
    "ChordModelConfig",
    "ChordRecognitionHeads",
    "ChordRecognitionLoss",
    "ChordRecognitionModel",
    "ClassificationHead",
    "ConformerBlock",
    "ConvSubsampling",
    "CQTNormalization",
    "CQTProcessor",
    "FocalLoss",
    "IDX_TO_CHORD_TYPE",
    "IDX_TO_PITCH_CLASS",
    "LARGE_CONFIG",
    "MEDIUM_CONFIG",
    "PITCH_CLASS_TO_IDX",
    "SMALL_CONFIG",
    "create_model",
    "format_chord_name",
    "get_config",
]
