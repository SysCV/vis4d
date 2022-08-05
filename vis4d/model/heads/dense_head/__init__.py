"""Dense heads."""
from .base import BaseSegmentationHead, BaseDenseBox2DHead
from .mmdet import MMBaseDenseHead
from .mmseg import MMSegDecodeHead

__all__ = [
    "BaseDenseBox2DHead",
    "BaseSegmentationHead",
    "MMSegDecodeHead",
    "MMBaseDenseHead",
]
