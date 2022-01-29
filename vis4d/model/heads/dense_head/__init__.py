"""Dense heads."""
from .base import BaseDenseHead, ClsDenseHead, DetDenseHead, SegDenseHead
from .mmcls import MMClsHead
from .mmdet import MMDetDenseHead, MMDetRPNHead
from .mmseg import MMSegDecodeHead
from .multi_cls_head import MultiClsHead

__all__ = [
    "BaseDenseHead",
    "ClsDenseHead",
    "DetDenseHead",
    "SegDenseHead",
    "MMSegDecodeHead",
    "MMDetDenseHead",
    "MMDetRPNHead",
    "MMClsHead",
    "MultiClsHead",
]
