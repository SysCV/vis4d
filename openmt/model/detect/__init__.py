"""Detector module."""
from .base import BaseDetector, BaseDetectorConfig, build_detector
from .d2_rcnn import D2GeneralizedRCNN

__all__ = [
    "BaseDetector",
    "D2GeneralizedRCNN",
    "BaseDetectorConfig",
    "build_detector",
]
