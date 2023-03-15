"""Evaluation protocols and metrics for different tasks."""

from .base import Evaluator
from .detect.coco import COCOEvaluator
from .occupancy import OccupancyEvaluator
from .segment import SegmentationEvaluator
from .classify import ClassificationEvaluator

__all__ = [
    "Evaluator",
    "ClassificationEvaluator",
    "COCOEvaluator",
    "SegmentationEvaluator",
    "OccupancyEvaluator",
    "ClassificationEvaluator",
]
