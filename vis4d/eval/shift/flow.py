"""SHIFT optical flow estimation evaluator."""

from __future__ import annotations

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
