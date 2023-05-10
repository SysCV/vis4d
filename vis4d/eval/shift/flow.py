"""SHIFT optical flow estimation evaluator."""
from __future__ import annotations

import numpy as np

from vis4d.common.typing import NDArrayNumber

from ..common import OpticalFlowEvaluator


class SHIFTOpticalFlowEvaluator(OpticalFlowEvaluator):
    """SHIFT optical flow estimation evaluation class."""

    def __init__(
        self,
    ) -> None:
        """Initialize the evaluator."""
        super().__init__(max_flow=200.0, use_degrees=False, scale=1.0)

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return "SHIFT Optical Flow Estimation Evaluator"

    @staticmethod
    def process(  # type: ignore # pylint: disable=arguments-differ
        self, prediction: NDArrayNumber, groundtruth: NDArrayI64
    ) -> None:
        """Process sample and update confusion matrix.

        Args:
            prediction: Predictions of shape (H, W, 2).
            groundtruth: Groundtruth of shape (H, W, 2).
        """
        super().process(prediction, groundtruth)
