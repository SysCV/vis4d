"""RoI heads."""
from .base import BaseRoIHead, RoIHeadConfig, build_roi_head
from .quasi_dense_embedding_head import QDRoIHead

__all__ = ["BaseRoIHead", "QDRoIHead", "build_roi_head", "RoIHeadConfig"]
