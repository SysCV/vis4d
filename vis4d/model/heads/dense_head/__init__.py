"""Dense heads."""
from .base import BaseDenseHead
from .mmdet import MMDetDenseHead, MMDetRPNHead
from .mmseg import MMSegDecodeHead

__all__ = [
    "BaseDenseHead",
    "MMSegDecodeHead",
    "MMDetDenseHead",
    "MMDetRPNHead",
]
