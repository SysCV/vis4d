"""Dense heads."""
from .base import BaseDenseHead
from .mmdet import MMDetDenseHead
from .mmseg import MMSegDecodeHead

__all__ = [
    "BaseDenseHead",
    "MMSegDecodeHead",
    "MMDetDenseHead",
]
