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
    return np.mean(np.sqrt(np.sum((prediction - target) ** 2, axis=-1))).item()


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
    return np.mean(
        np.arccos(
            np.clip(
                np.abs(np.sum(prediction * target, axis=-1))
                / (
                    np.linalg.norm(prediction, axis=-1)
                    * np.linalg.norm(target, axis=-1)
                    + epsilon
                ),
                0.0,
                1.0,
            )
        )
    ).item()
