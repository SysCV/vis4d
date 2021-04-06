"""Meta architectures for tracking."""

from .base import BaseMetaArch, build_model
from .quasi_dense_rcnn import QDGeneralizedRCNN

__all__ = ["QDGeneralizedRCNN", "BaseMetaArch", "build_model"]
