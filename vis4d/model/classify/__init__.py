"""Classifier module."""
from .base import BaseClassifier
from .mmcls import MMImageClassifier
from .multi_label import MultiLabelClassifier

__all__ = ["BaseClassifier", "MMImageClassifier", "MultiLabelClassifier"]
