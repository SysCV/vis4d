"""SHIFT depth estimation evaluator."""
from __future__ import annotations

import numpy as np

from vis4d.common.typing import NDArrayNumber

from ..common import DepthEvaluator


def apply_garg_crop(mask: NDArrayNumber) -> NDArrayNumber:
    """Apply Garg ECCV16 crop to the mask.

    Args:
        mask (np.array): Mask to be cropped, in shape (N, H, W).

    Returns:
        np.array: Cropped mask, in shape (N, H', W').
    """
    # crop used by Garg ECCV16
    h, w = mask.shape
    crop = np.array(
        [0.40810811 * h, 0.99189189 * h, 0.03594771 * w, 0.96405229 * w]
    ).astype(np.int32)
    return mask[:, crop[0] : crop[1], crop[2] : crop[3]]


def apply_eigen_crop(mask: NDArrayNumber) -> NDArrayNumber:
    """Apply Eigen NIPS14 crop to the mask.

    Args:
        mask (np.array): Mask to be cropped, in shape (N, H, W).

    Returns:
        np.array: Cropped mask, in shape (N, H', W').
    """
    # https://github.com/mrharicot/monodepth/utils/evaluate_kitti.py
    h, w = mask.shape
    crop = np.array(
        [0.3324324 * h, 0.91351351 * h, 0.0359477 * w, 0.96405229 * w]
    ).astype(np.int32)
    return mask[:, crop[0] : crop[1], crop[2] : crop[3]]


class SHIFTDepthEvaluator(DepthEvaluator):
    """SHIFT depth estimation evaluation class."""

    def __init__(self, evaluation_crop: str | None = None) -> None:
        """Initialize the evaluator.

        Args:
            evaluation_crop (str, optional): Evaluation crop preset, either
                "garg" or "eigen". Defaults to None, which means no crop.
        """
        super().__init__(min_depth=0.05, max_depth=80.0)
        assert evaluation_crop in {"garg", "eigen", None}, (
            f"Invalid evaluation crop {evaluation_crop}. "
            "Supported options are 'garg' and 'eigen'."
        )
        self.evaluation_crop = evaluation_crop

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return "SHIFT Depth Estimation Evaluator"

    def process(  # type: ignore # pylint: disable=arguments-differ
        self, prediction: NDArrayNumber, groundtruth: NDArrayNumber
    ) -> None:
        """Process sample and update confusion matrix.

        Args:
            prediction: Predictions of shape (N, H, W).
            groundtruth: Groundtruth of shape (N, H, W).
        """
        if self.evaluation_crop == "garg":
            prediction = apply_garg_crop(prediction)
            groundtruth = apply_garg_crop(groundtruth)
        elif self.evaluation_crop == "eigen":
            prediction = apply_eigen_crop(prediction)
            groundtruth = apply_eigen_crop(groundtruth)
        super().process(prediction, groundtruth)
