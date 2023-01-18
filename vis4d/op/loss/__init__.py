"""This module contains commonly used loss functions.

The losses do not follow a common API, but have a reducer as attribute,
which is a function to aggregate loss values into a single tensor value.
"""

from .base import Loss
from .box3d_uncertainty_loss import Box3DUncertaintyLoss
from .embedding_distance import EmbeddingDistanceLoss
from .multi_pos_cross_entropy import MultiPosCrossEntropyLoss
from .orthogonal_transform_loss import OrthogonalTransformRegularizationLoss

__all__ = [
    "Loss",
    "EmbeddingDistanceLoss",
    "MultiPosCrossEntropyLoss",
    "Box3DUncertaintyLoss",
    "OrthogonalTransformRegularizationLoss",
]
