"""Evaluation protocols and metrics for different tasks."""

from .base import Evaluator
from .coco import COCOEvaluator

__all__ = ["Evaluator", "COCOEvaluator"]
