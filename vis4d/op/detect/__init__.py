"""Detector module."""
from .base import BaseDetector, BaseOneStageDetector, BaseTwoStageDetector
from .faster_rcnn import FasterRCNNHead

__all__ = [
    "BaseDetector",
    "FasterRCNNHead",
]
