"""Depth estimation metrics."""

from __future__ import annotations

import numpy as np

from vis4d.common.typing import ArrayLike

from ..utils import check_shape_match, dense_inputs_to_numpy


def end_point_error(prediction: ArrayLike, target: ArrayLike) -> float:
    """Compute the end point error.

    Args:
        prediction (ArrayLike): Prediction UV optical flow, in shape (..., 2).
        target (ArrayLike): Target UV optical flow, in shape (..., 2).

    Returns:
        float: End point error.
    """
    prediction, target = dense_inputs_to_numpy(prediction, target)
    check_shape_match(prediction, target)
    squared_sum = np.sum((prediction - target) ** 2, axis=-1)
    return np.mean(np.sqrt(squared_sum)).item()


def angular_error(
    prediction: ArrayLike, target: ArrayLike, epsilon: float = 1e-6
) -> float:
    """Compute the angular error.

    Args:
        prediction (ArrayLike): Prediction UV optical flow, in shape (..., 2).
        target (ArrayLike): Target UV optical flow, in shape (..., 2).
        epsilon (float, optional): Epsilon value for numerical stability.

    Returns:
        float: Angular error.
    """
    prediction, target = dense_inputs_to_numpy(prediction, target)
    check_shape_match(prediction, target)
    product = np.sum(prediction * target, axis=-1)
    pred_norm = np.linalg.norm(prediction, axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)
    cos_angle = np.abs(product) / (pred_norm * target_norm + epsilon)
    return np.mean(np.arccos(np.clip(cos_angle, 0.0, 1.0))).item()
