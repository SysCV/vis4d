"""Utility functions for evaluation."""

import numpy as np

from vis4d.common.array import array_to_numpy
from vis4d.common.typing import ArrayLike, NDArrayNumber


def apply_garg_crop(mask: NDArrayNumber) -> NDArrayNumber:
    """Apply Garg ECCV16 crop to the mask.

    Args:
        mask (np.array): Mask to be cropped, in shape (..., H, W).

    Returns:
        np.array: Cropped mask, in shape (..., H', W').
    """
    # crop used by Garg ECCV16
    h, w = mask.shape[-2:]
    crop = np.array(
        [0.40810811 * h, 0.99189189 * h, 0.03594771 * w, 0.96405229 * w]
    ).astype(np.int32)
    return mask[..., crop[0] : crop[1], crop[2] : crop[3]]


def apply_eigen_crop(mask: NDArrayNumber) -> NDArrayNumber:
    """Apply Eigen NIPS14 crop to the mask.

    Args:
        mask (np.array): Mask to be cropped, in shape (N, H, W).

    Returns:
        np.array: Cropped mask, in shape (N, H', W').
    """
    # https://github.com/mrharicot/monodepth/utils/evaluate_kitti.py
    h, w = mask.shape[-2:]
    crop = np.array(
        [0.3324324 * h, 0.91351351 * h, 0.0359477 * w, 0.96405229 * w]
    ).astype(np.int32)
    return mask[..., crop[0] : crop[1], crop[2] : crop[3]]


def dense_inputs_to_numpy(
    prediction: ArrayLike, target: ArrayLike
) -> tuple[NDArrayNumber, NDArrayNumber]:
    """Convert dense prediction and target to numpy arrays."""
    prediction = array_to_numpy(prediction, n_dims=None, dtype=np.float32)
    target = array_to_numpy(target, n_dims=None, dtype=np.float32)
    return prediction, target


def check_shape_match(
    prediction: NDArrayNumber, target: NDArrayNumber
) -> None:
    """Check if the shape of prediction and target matches."""
    assert prediction.shape == target.shape, (
        f"Shape mismatch between prediction {prediction.shape} and target"
        f"{target.shape}."
    )
