"""SHIFT depth estimation evaluator."""

from __future__ import annotations

from vis4d.common.typing import NDArrayNumber

from ..common import DepthEvaluator


def apply_crop(depth: NDArrayNumber) -> NDArrayNumber:
    """Apply crop to depth map to match SHIFT evaluation."""
    return depth[..., 0:740, :]


class SHIFTDepthEvaluator(DepthEvaluator):
    """SHIFT depth estimation evaluation class."""

    def __init__(self, use_eval_crop: bool = True) -> None:
        """Initialize the evaluator.

        Args:
            use_eval_crop (bool): Whether to use the evaluation crop.
                Default: True.
        """
        super().__init__(min_depth=0.01, max_depth=80.0)
        self.use_eval_crop = use_eval_crop

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return "SHIFT Depth Estimation Evaluator"

    def process_batch(  # type: ignore # pylint: disable=arguments-differ
        self, prediction: NDArrayNumber, groundtruth: NDArrayNumber
    ) -> None:
        """Process sample and update confusion matrix.

        Args:
            prediction: Predictions of shape (N, H, W).
            groundtruth: Groundtruth of shape (N, H, W).
        """
        if self.use_eval_crop:
            prediction = apply_crop(prediction)
            groundtruth = apply_crop(groundtruth)
        print(prediction.shape, groundtruth.shape)
        super().process_batch(prediction, groundtruth)
