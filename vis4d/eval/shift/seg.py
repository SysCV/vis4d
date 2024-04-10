"""SHIFT segmentation evaluator."""

from __future__ import annotations

from vis4d.common.typing import NDArrayI64, NDArrayNumber
from vis4d.data.datasets.shift import shift_seg_ignore, shift_seg_map
from vis4d.eval.common.seg import SegEvaluator


class SHIFTSegEvaluator(SegEvaluator):
    """SHIFT segmentation evaluation class."""

    inverse_seg_map = {v: k for k, v in shift_seg_map.items()}

    def __init__(self, ignore_classes_as_cityscapes: bool = True) -> None:
        """Initialize the evaluator."""
        super().__init__(
            num_classes=23,
            class_to_ignore=255,
            class_mapping=self.inverse_seg_map,
        )
        self.ignore_classes_as_cityscapes = ignore_classes_as_cityscapes

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return "SHIFT Segmentation Evaluator"

    def _prune_class(self, label: NDArrayI64) -> NDArrayI64:
        """Prune class labels."""
        for cls in shift_seg_ignore:
            label[label == shift_seg_map[cls]] = 255
        return label

    def process_batch(  # type: ignore  # pylint: disable=arguments-differ
        self, prediction: NDArrayNumber, groundtruth: NDArrayI64
    ) -> None:
        """Process sample and update confusion matrix.

        Args:
             prediction: Predictions of shape [N,C,...] or [N,...] with
                    C* being any number if channels. Note, C is passed,
                    the prediction is converted to target labels by applying
                    the max operations along the second axis
             groundtruth: Groundtruth of shape [N_batch, ...] type int
        """
        if self.ignore_classes_as_cityscapes:
            groundtruth = self._prune_class(groundtruth)
        super().process_batch(prediction, groundtruth)
