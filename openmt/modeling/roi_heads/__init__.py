"""RoI heads."""


from .base_roi_head import BaseRoIHead
from .quasi_dense_embedding_head import QDRoIHead

__all__ = ["BaseRoIHead", "QDRoIHead"]
