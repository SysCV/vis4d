"""OpenMT loss implementations."""

from .base_loss import BaseLoss, LossConfig, build_loss
from .embedding_distance_loss import EmbeddingDistanceLoss
from .multi_pos_cross_entropy_loss import MultiPosCrossEntropyLoss

__all__ = [
    "BaseLoss",
    "build_loss",
    "LossConfig",
    "EmbeddingDistanceLoss",
    "MultiPosCrossEntropyLoss",
]
