"""Dense heads."""
from .base import BaseDenseHead, BaseDenseHeadConfig, build_dense_head
from .mmdet_wrapper import MMDetDenseHead, MMDetDenseHeadConfig
from .mmseg_wrapper import MMSegDecodeHead, MMSegDecodeHeadConfig

__all__ = [
    "BaseDenseHead",
    "BaseDenseHeadConfig",
    "build_dense_head",
    "MMSegDecodeHead",
    "MMSegDecodeHeadConfig",
    "MMDetDenseHead",
    "MMDetDenseHeadConfig",
]
