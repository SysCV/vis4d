"""Classifier module."""
from .base import BaseClassifier
from .mmcls import MMImageClassifier
from .multi_cls import MultiImageClassifier

__all__ = ["BaseClassifier", "MMImageClassifier", "MultiImageClassifier"]
