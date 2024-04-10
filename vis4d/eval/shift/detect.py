"""SHIFT detection evaluator."""

from __future__ import annotations

from vis4d.data.datasets.shift import shift_det_map

from ..scalabel import ScalabelDetectEvaluator


class SHIFTDetectEvaluator(ScalabelDetectEvaluator):
    """SHIFT detection evaluation class."""

    inverse_det_map = {v: k for k, v in shift_det_map.items()}

    def __init__(self, annotation_path: str) -> None:
        """Initialize the evaluator."""
        super().__init__(annotation_path=annotation_path, mask_threshold=0)
        self.inverse_cat_map = self.inverse_det_map
