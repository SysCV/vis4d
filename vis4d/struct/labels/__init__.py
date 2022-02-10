"""Vis4D labels module."""
from .boxes import Boxes2D, Boxes3D
from .masks import InstanceMasks, MaskLogits, Masks, SemanticMasks, TMasks

__all__ = [
    "Boxes2D",
    "Boxes3D",
    "Masks",
    "TMasks",
    "MaskLogits",
    "InstanceMasks",
    "SemanticMasks",
]
