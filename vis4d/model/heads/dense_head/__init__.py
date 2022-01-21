"""Dense heads."""
from .base import BaseDenseHead, DetDenseHead, SegDenseHead
from .mmdet import MMDetDenseHead, MMDetRPNHead
from .mmseg import MMSegDecodeHead

__all__ = [
    "BaseDenseHead",
    "DetDenseHead",
    "SegDenseHead",
    "MMSegDecodeHead",
    "MMDetDenseHead",
    "MMDetRPNHead",
]
