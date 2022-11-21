"""Evaluation protocols and metrics for different tasks."""

from .base import Evaluator
from .detect.coco import COCOEvaluator
from .occupancy import OccupancyEvaluator
from .segment import SegmentationEvaluator

__all__ = [
    "Evaluator",
    "COCOEvaluator",
    "SegmentationEvaluator",
    "OccupancyEvaluator",
]
