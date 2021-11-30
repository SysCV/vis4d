"""Dense heads."""
from .base import BaseDenseHead, BaseDenseHeadConfig, build_dense_head
from .mmseg_wrapper import MMDecodeHead

__all__ = [
    "BaseDenseHead",
    "BaseDenseHeadConfig",
    "build_dense_head",
    "MMDecodeHead",
]
