"""Init sampler module."""
from .base import BaseRoIPooler
from .roi_pooler import MultiScaleRoIPooler

__all__ = [
    "BaseRoIPooler",
    "MultiScaleRoIPooler",
]
