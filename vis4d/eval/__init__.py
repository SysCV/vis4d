"""Evaluation package."""

from .base import Evaluator
from .coco import COCOEvaluator

__all__ = ["Evaluator", "COCOEvaluator"]
