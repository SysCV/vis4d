"""Vis4D tracking loss implementations."""

from .base import Loss
from .box3d_uncertainty_loss import Box3DUncertaintyLoss
from .embedding_distance import EmbeddingDistanceLoss
from .multi_pos_cross_entropy import MultiPosCrossEntropyLoss

__all__ = [
    "Loss",
    "EmbeddingDistanceLoss",
    "MultiPosCrossEntropyLoss",
    "Box3DUncertaintyLoss",
]
