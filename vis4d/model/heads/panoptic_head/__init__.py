"""Panoptic heads."""
from .base import BasePanopticHead, BasePanopticHeadConfig, build_panoptic_head
from .simple import SimplePanopticHead, SimplePanopticHeadConfig

__all__ = [
    "BasePanopticHeadConfig",
    "BasePanopticHead",
    "build_panoptic_head",
    "SimplePanopticHead",
    "SimplePanopticHeadConfig",
]
