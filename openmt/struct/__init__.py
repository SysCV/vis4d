"""OpenMT struct module."""
from .structures import (
    Boxes2D,
    DetectionOutput,
    ImageList,
    Instances,
    TorchCheckpoint,
)

__all__ = [
    "Boxes2D",
    "Instances",
    "TorchCheckpoint",
    "ImageList",
    "DetectionOutput",
]
