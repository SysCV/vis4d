"""SHIFT tracking evaluator."""

from __future__ import annotations

from vis4d.data.datasets.shift import shift_det_map

from ..scalabel import ScalabelTrackEvaluator


class SHIFTTrackEvaluator(ScalabelTrackEvaluator):
    """SHIFT tracking evaluation class."""

    inverse_det_map = {v: k for k, v in shift_det_map.items()}

    def __init__(
        self, annotation_path: str, mask_threshold: float = 0.0
    ) -> None:
        """Initialize the evaluator."""
        super().__init__(
            annotation_path=annotation_path, mask_threshold=mask_threshold
        )
        self.inverse_cat_map = self.inverse_det_map
