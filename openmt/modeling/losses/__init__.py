"""OpenMT loss implementations."""

from .base import BaseLoss, LossConfig, build_loss
from .embedding_distance import EmbeddingDistanceLoss
from .multi_pos_cross_entropy import MultiPosCrossEntropyLoss

__all__ = [
    "BaseLoss",
    "build_loss",
    "LossConfig",
    "EmbeddingDistanceLoss",
    "MultiPosCrossEntropyLoss",
]
