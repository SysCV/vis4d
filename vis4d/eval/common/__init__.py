"""Common evaluation code."""

from .binary import BinaryEvaluator
from .cls import ClassificationEvaluator
from .depth import DepthEvaluator
from .flow import OpticalFlowEvaluator
from .seg import SegEvaluator

__all__ = [
    "ClassificationEvaluator",
    "DepthEvaluator",
    "OpticalFlowEvaluator",
    "BinaryEvaluator",
    "SegEvaluator",
]
