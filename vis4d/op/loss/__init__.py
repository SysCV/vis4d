"""This module contains commonly used loss functions.

The losses do not follow a common API, but have a reducer as attribute,
which is a function to aggregate loss values into a single tensor value.
"""

from .base import Loss
from .embedding_distance import EmbeddingDistanceLoss
from .iou_loss import IoULoss
from .multi_level_seg_loss import MultiLevelSegLoss
from .multi_pos_cross_entropy import MultiPosCrossEntropyLoss
from .orthogonal_transform_loss import OrthogonalTransformRegularizationLoss
from .seg_cross_entropy_loss import SegCrossEntropyLoss

__all__ = [
    "Loss",
    "EmbeddingDistanceLoss",
    "IoULoss",
    "MultiLevelSegLoss",
    "MultiPosCrossEntropyLoss",
    "OrthogonalTransformRegularizationLoss",
    "SegCrossEntropyLoss",
]
