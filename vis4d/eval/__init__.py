"""Evaluation protocols and metrics for different tasks."""

from .base import Evaluator
from .classify import ClassificationEvaluator
from .detect.coco import COCOEvaluator
from .occupancy import OccupancyEvaluator
from .seg import BDD100KSegEvaluator, SegEvaluator

__all__ = [
    "Evaluator",
    "ClassificationEvaluator",
    "COCOEvaluator",
    "BDD100KSegEvaluator",
    "SegEvaluator",
    "OccupancyEvaluator",
    "ClassificationEvaluator",
]
