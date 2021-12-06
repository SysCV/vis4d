"""Dense heads."""
from .base import BaseDenseHead, BaseDenseHeadConfig, build_dense_head
from .mmdet import MMDetDenseHead, MMDetDenseHeadConfig
from .mmseg import MMSegDecodeHead, MMSegDecodeHeadConfig

__all__ = [
    "BaseDenseHead",
    "BaseDenseHeadConfig",
    "build_dense_head",
    "MMSegDecodeHead",
    "MMSegDecodeHeadConfig",
    "MMDetDenseHead",
    "MMDetDenseHeadConfig",
]
