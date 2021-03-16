"""Meta architectures for tracking."""

from .base_arch import BaseMetaArch
from .quasi_dense_rcnn import QDGeneralizedRCNN

__all__ = ["QDGeneralizedRCNN", "BaseMetaArch"]
