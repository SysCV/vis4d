"""Dense heads."""
from .base import BaseDenseBox2DHead, BaseSegmentationHead
from .mmdet import MMDetDenseHead
from .mmseg import MMSegDecodeHead

__all__ = [
    "BaseDenseBox2DHead",
    "BaseSegmentationHead",
    "MMSegDecodeHead",
    "MMDetDenseHead",
]
