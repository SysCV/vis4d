"""KITTI evaluation code."""

from __future__ import annotations

import numpy as np

from vis4d.common.typing import NDArrayFloat, NDArrayNumber

from ..common import DepthEvaluator


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
    mask[..., crop[0] : crop[1], crop[2] : crop[3]] = 1
    return mask


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
    mask[..., crop[0] : crop[1], crop[2] : crop[3]] = 1
    return mask


class KITTIDepthEvaluator(DepthEvaluator):
    """KITTI depth evaluation class."""

    METRIC_DEPTH = "depth"

    def __init__(
        self,
        min_depth: float = 0.01,
        max_depth: float = 80.0,
        eval_crop: str | None = None,
    ) -> None:
        """Initialize KITTI depth evaluator."""
        super().__init__(min_depth, max_depth)
        self.eval_crop = eval_crop
        self.reset()

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return "KITTI evaluation for depth"

    def _get_eval_mask(self, valid_mask: NDArrayNumber) -> NDArrayNumber:
        """Do Grag or Eigen cropping for testing."""
        eval_mask = np.zeros_like(valid_mask)
        if self.eval_crop == "garg_crop":
            eval_mask = apply_garg_crop(eval_mask)
        elif self.eval_crop == "eigen_crop":
            eval_mask = apply_eigen_crop(eval_mask)
        else:
            eval_mask = np.ones_like(valid_mask)
        return np.logical_and(valid_mask, eval_mask)

    def _apply_mask(
        self, prediction: NDArrayFloat, target: NDArrayFloat
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Apply mask to prediction and target."""
        valid_mask = (target > self.min_depth) & (target < self.max_depth)
        eval_mask = self._get_eval_mask(valid_mask)
        prediction = prediction[eval_mask]
        target = target[eval_mask]
        return prediction, target
