"""Evaluates Binary Predictions."""
from typing import List, Tuple

import torch

from vis4d.common import MetricLogs, ModelOutput
from vis4d.data.const import COMMON_KEYS
from vis4d.data.datasets.base import DictData

from .base import Evaluator


class BinaryEvaluator(Evaluator):
    """Creates an Evaluator that evaluates binary predictions."""

    METRIC_IOU = "IoU"
    METRIC_ALL = "all"
    # TODO create accuracy, f1, ....
    # TODO write tests
    def __init__(
        self,
        binary_prediction_key: str = COMMON_KEYS.occupancy3d,
        binary_gt_key: str = COMMON_KEYS.occupancy3d,
        threshold=0.5,
    ):
        """Creates a new binary evaluator.

        Args:
            binary_prediction_key (str): Key for the prediction data
            binary_gt_key (str): Key for the groundtruth data
            threshold (float): Threshold for prediction to convert
                               to binary.
        """
        super().__init__()
        self.binary_prediction_key = binary_prediction_key
        self.binary_gt_key = binary_gt_key
        self.threshold = threshold
        self.reset()
        self.iou_scores: List[float] = []

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

    @property
    def metrics(self) -> List[str]:
        """Supported metrics."""
        return [BinaryEvaluator.METRIC_IOU]

    def reset(self) -> None:
        """Reset the saved predictions to start new round of evaluation."""
        self.iou_scores = []

    def process(self, inputs: DictData, outputs: ModelOutput) -> None:
        """Processes a new (batch) of predictions."""
        targets = inputs[self.binary_gt_key]
        predictions = outputs[self.binary_prediction_key]

        # Calculate miou
        self.iou_scores.append(self._iou(predictions, targets))

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
