"""Depth estimation metrics."""

from __future__ import annotations

import numpy as np

from vis4d.common.typing import ArrayLike

from ..utils import check_shape_match, dense_inputs_to_numpy


def absolute_error(prediction: ArrayLike, target: ArrayLike) -> float:
    """Compute the absolute error.

    Args:
        prediction (NDArrayNumber): Prediction depth map, in shape (..., H, W).
        target (NDArrayNumber): Target depth map, in shape (..., H, W).

    Returns:
        float: Absolute error.
    """
    prediction, target = dense_inputs_to_numpy(prediction, target)
    check_shape_match(prediction, target)
    return np.mean(np.abs(prediction - target))


def squared_relative_error(prediction: ArrayLike, target: ArrayLike) -> float:
    """Compute the squared relative error.

    Args:
        prediction (NDArrayNumber): Prediction depth map, in shape (..., H, W).
        target (NDArrayNumber): Target depth map, in shape (..., H, W).

    Returns:
        float: Square relative error.
    """
    prediction, target = dense_inputs_to_numpy(prediction, target)
    check_shape_match(prediction, target)
    return np.mean(np.square(prediction - target) / np.square(target))


def absolute_relative_error(prediction: ArrayLike, target: ArrayLike) -> float:
    """Compute the absolute relative error.

    Args:
        prediction (NDArrayNumber): Prediction depth map, in shape (..., H, W).
        target (NDArrayNumber): Target depth map, in shape (..., H, W).

    Returns:
        float: Absolute relative error.
    """
    prediction, target = dense_inputs_to_numpy(prediction, target)
    check_shape_match(prediction, target)
    return np.mean(np.abs(prediction - target) / target)


def root_mean_squared_error(prediction: ArrayLike, target: ArrayLike) -> float:
    """Compute the root mean squared error.

    Args:
        prediction (ArrayLike): Prediction depth map, in shape (..., H, W).
        target (ArrayLike): Target depth map, in shape (..., H, W).

    Returns:
        float: Root mean squared error.
    """
    prediction, target = dense_inputs_to_numpy(prediction, target)
    check_shape_match(prediction, target)
    return np.sqrt(np.mean(np.square(prediction - target)))


def root_mean_squared_error_log(
    prediction: ArrayLike, target: ArrayLike, epsilon: float = 1e-8
) -> float:
    """Compute the root mean squared error in log space.

    Args:
        prediction (ArrayLike): Prediction depth map, in shape (H, W).
        target (ArrayLike): Target depth map, in shape (H, W).
        epsilon (float, optional): Epsilon to avoid log(0). Defaults to 1e-6.

    Returns:
        float: Root mean squared error in log space.
    """
    prediction, target = dense_inputs_to_numpy(prediction, target)
    check_shape_match(prediction, target)
    return np.sqrt(
        np.mean(
            np.square(np.log(prediction + epsilon) - np.log(target + epsilon))
        )
    )


def scale_invariant_log(
    prediction: ArrayLike, target: ArrayLike, epsilon: float = 1e-8
) -> float:
    """Compute the scale invariant log error.

    Args:
        prediction (ArrayLike): Prediction depth map, in shape (H, W).
        target (ArrayLike): Target depth map, in shape (H, W).
        epsilon (float, optional): Epsilon to avoid log(0). Defaults to 1e-6.

    Returns:
        float: Scale invariant log error.
    """
    prediction, target = dense_inputs_to_numpy(prediction, target)
    check_shape_match(prediction, target)
    return (
        100
        * np.sqrt(
            np.var(np.log(prediction + epsilon) - np.log(target + epsilon))
        ).mean(),
    )


def delta_p(
    prediction: ArrayLike, target: ArrayLike, power: float = 1
) -> float:
    """Compute the delta_p metric.

    Args:
        prediction (ArrayLike): Prediction depth map, in shape (H, W).
        target (ArrayLike): Target depth map, in shape (H, W).
        power (float, optional): Power of the threshold. Defaults to 1.

    Returns:
        float: Delta_p metric.
    """
    prediction, target = dense_inputs_to_numpy(prediction, target)
    check_shape_match(prediction, target)
    return (
        np.maximum((target / prediction), (prediction / target))
        < 1.25**power
    )


def log_10_error(prediction: ArrayLike, target: ArrayLike) -> float:
    """Compute the log_10 error.

    Args:
        prediction (ArrayLike): Prediction depth map, in shape (H, W).
        target (ArrayLike): Target depth map, in shape (H, W).

    Returns:
        float: Log_10 error.
    """
    prediction, target = dense_inputs_to_numpy(prediction, target)
    check_shape_match(prediction, target)
    return np.mean(np.abs(np.log10(prediction) - np.log10(target)))
