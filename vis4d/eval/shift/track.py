"""SHIFT tracking evaluator."""
from __future__ import annotations

from vis4d.data.datasets.shift import shift_det_map

from ..scalabel import ScalabelTrackEvaluator


class SHIFTTrackEvaluator(ScalabelTrackEvaluator):
    """SHIFT tracking evaluation class."""

    inverse_det_map = {v: k for k, v in shift_det_map.items()}

    def __init__(self) -> None:
        """Initialize the evaluator."""
        super().__init__(mask_threshold=0)
        self.inverse_cat_map = self.inverse_det_map