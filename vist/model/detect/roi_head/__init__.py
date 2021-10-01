"""RoI heads."""
from .base import BaseRoIHead, BaseRoIHeadConfig, build_roi_head
<<<<<<< HEAD
from .qd_3dt_bbox_3d_head import QD3DTBBox3DHead
=======
from .qd_3dt_bbox3d_head import QD3DTBBox3DHead
>>>>>>> main

__all__ = [
    "BaseRoIHeadConfig",
    "BaseRoIHead",
    "build_roi_head",
    "QD3DTBBox3DHead",
]
