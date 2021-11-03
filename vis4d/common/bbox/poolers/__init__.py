"""Init sampler module."""
from .base import BaseRoIPooler, RoIPoolerConfig, build_roi_pooler
from .roi_pooler import MultiScaleRoIPooler

__all__ = [
    "BaseRoIPooler",
    "MultiScaleRoIPooler",
    "build_roi_pooler",
    "RoIPoolerConfig",
]
