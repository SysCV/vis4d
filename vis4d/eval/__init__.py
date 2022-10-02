"""Evaluation package."""

from .base import BaseEvaluator
from .coco import COCOEvaluator

__all__ = ["BaseEvaluator", "COCOEvaluator"]
