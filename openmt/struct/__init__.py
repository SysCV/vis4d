"""OpenMT struct module."""
from .data import (
    EvalResult,
    EvalResults,
    Images,
    InputSample,
    LossesType,
    TorchCheckpoint,
)
from .labels import Boxes2D, DetectionOutput, LabelInstance, ModelOutput

__all__ = [
    "Boxes2D",
    "LabelInstance",
    "TorchCheckpoint",
    "LossesType",
    "EvalResult",
    "EvalResults",
    "InputSample",
    "Images",
    "DetectionOutput",
    "ModelOutput",
]
