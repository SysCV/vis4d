"""BinaryEvaluator."""
from typing import Dict, List, Tuple

import torch

from vis4d.common import MetricLogs, ModelOutput
from vis4d.common.typing import COMMON_KEYS
from vis4d.data.datasets.base import DictData

from .base import Evaluator


class BinaryEvaluator(Evaluator):
    """Creates an evaluation to reports mIoU score and confusion matrix."""

    METRIC_IOU = "IoU"
    # METRIC_ACCURACY = "accuracy" #TODO
    # METRIC_F1 = "f1" #TODO
    METRIC_ALL = "all"

    def __init__(
        self,
        binary_prediction_key: str = COMMON_KEYS.occupancy3d,
        binary_gt_key: str = COMMON_KEYS.occupancy3d,
        threshold=0.5,
    ):
        """Creates a new evaluator.

        Args:
            num_classes (int): Number of semantic classes
            class_mapping (int): Dict mapping each class_id to a readable name
            binary_prediction_key (str): Key to obtain the occupancy from the model output
            binary_gt_key (str): Key to obtain the gt occupancy from the input struct

        """
        super().__init__()
        self.binary_prediction_key = binary_prediction_key
        self.binary_gt_key = binary_gt_key
        self.threshold = threshold
        self.reset()

        self.iou_scores = []

    def _iou(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        prediction_bin = prediction > self.threshold
        target_bin = target > self.threshold

        # Put all data in second dimension
        # Also works for 1-dimensional data
        if prediction_bin.ndim >= 2:
            prediction_bin = prediction_bin.ravel()
        if target_bin.ndim >= 2:
            target_bin = target_bin.ravel()

        # Compute IOU
        area_union = (target_bin | prediction_bin).sum()
        area_intersect = (target_bin & prediction_bin).sum()

        iou = area_intersect / area_union

        return iou.item()

    # @property
    def metrics(self) -> List[str]:
        """Supported metrics."""
        return [BinaryEvaluator.METRIC_IOU]

    def reset(self) -> None:
        """Reset the saved predictions to start new round of evaluation."""
        # self._confusion_matrix.reset()
        self.iou_scores = []

    def process(self, inputs: DictData, outputs: ModelOutput) -> None:
        """TODO."""
        targets = inputs[self.binary_gt_key]
        predictions = outputs[self.binary_prediction_key]

        # Calculate miou
        self.iou_scores.append(self._iou(predictions, targets))

    def _get_class_name_for_idx(self, idx: int) -> str:
        return self.class_mapping.get(idx, f"class_{idx}")

    def evaluate(self, metric: str) -> Tuple[MetricLogs, str]:
        """Evaluate predictions. Returns a dict containing the raw data and a
        short description string containing a readable result.

        Args:
            metric (str): Metric to use. See @property metric
        """
        metric_data, short_description = {}, ""
        if metric in [BinaryEvaluator.METRIC_IOU, BinaryEvaluator.METRIC_ALL]:
            metric_data["iou"] = sum(self.iou_scores) / len(self.iou_scores)
            short_description += "Iou: " + str(metric_data["iou"]) + "\n"

        return metric_data, short_description
