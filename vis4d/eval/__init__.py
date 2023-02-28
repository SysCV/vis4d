"""Evaluation protocols and metrics for different tasks."""

from .base import Evaluator
from .classification import ClassificationEvaluator
from .detect.coco import COCOEvaluator
from .occupancy import OccupancyEvaluator
from .segment import SegmentationEvaluator

__all__ = [
    "Evaluator",
    "ClassificationEvaluator",
    "COCOEvaluator",
    "SegmentationEvaluator",
    "OccupancyEvaluator",
    "ClassificationEvaluator",
]
