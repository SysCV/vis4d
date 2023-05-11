"""SHIFT depth estimation evaluator."""
from __future__ import annotations

from vis4d.common.typing import NDArrayNumber

from ..common import DepthEvaluator
from ..common.utils import apply_eigen_crop, apply_garg_crop


class SHIFTDepthEvaluator(DepthEvaluator):
    """SHIFT depth estimation evaluation class."""

    def __init__(self, evaluation_crop: str | None = None) -> None:
        """Initialize the evaluator.

        Args:
            evaluation_crop (str, optional): Evaluation crop preset, either
                "garg" or "eigen". Defaults to None, which means no crop.
        """
        super().__init__(min_depth=0.01, max_depth=80.0)
        assert evaluation_crop in {"garg", "eigen", None}, (
            f"Invalid evaluation crop {evaluation_crop}. "
            "Supported options are 'garg' and 'eigen'."
        )
        self.evaluation_crop = evaluation_crop

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
        if self.evaluation_crop == "garg":
            prediction = apply_garg_crop(prediction)
            groundtruth = apply_garg_crop(groundtruth)
        elif self.evaluation_crop == "eigen":
            prediction = apply_eigen_crop(prediction)
            groundtruth = apply_eigen_crop(groundtruth)
        super().process_batch(prediction, groundtruth)
