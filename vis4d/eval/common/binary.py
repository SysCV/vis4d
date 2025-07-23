"""Binary occupancy evaluator."""

from __future__ import annotations

import numpy as np

from vis4d.common.array import array_to_numpy
from vis4d.common.typing import (
    ArrayLike,
    MetricLogs,
    NDArrayBool,
    NDArrayNumber,
)
from vis4d.eval.base import Evaluator


def threshold_and_flatten(
    prediction: NDArrayNumber, target: NDArrayNumber, threshold_value: float
) -> tuple[NDArrayBool, NDArrayBool]:
    """Thresholds the predictions based on the provided treshold value.

    Applies the following actions:
        prediction -> prediction >= threshold_value
        pred, gt = pred.ravel().bool(), gt.ravel().bool()

    Args:
        prediction: Prediction array with continuous values
        target: Grondgtruth values {0,1}
        threshold_value: Value to use to convert the continuous prediction
                         into binary.

    Returns:
        tuple of two boolean arrays, prediction and target
    """
    prediction_bin: NDArrayBool = prediction >= threshold_value
    return prediction_bin.ravel().astype(bool), target.ravel().astype(bool)


class BinaryEvaluator(Evaluator):
    """Creates a new Evaluater that evaluates binary predictions."""

    METRIC_BINARY = "BinaryCls"

    KEY_IOU = "IoU"
    KEY_ACCURACY = "Accuracy"
    KEY_F1 = "F1"
    KEY_PRECISION = "Precision"
    KEY_RECALL = "Recall"

    def __init__(
        self,
        threshold: float = 0.5,
    ) -> None:
        """Creates a new binary evaluator.

        Args:
            threshold (float): Threshold for prediction to convert
                               to binary. All prediction that are higher than
                               this value will be assigned the 'True' label
        """
        super().__init__()
        self.threshold = threshold
        self.reset()

        self.true_positives: list[float] = []
        self.false_positives: list[float] = []
        self.true_negatives: list[float] = []
        self.false_negatives: list[float] = []
        self.n_samples: list[float] = []

        self.has_samples = False

    def _calc_confusion_matrix(
        self, prediction: NDArrayBool, target: NDArrayBool
    ) -> None:
        """Calculates the confusion matrix and stores them as attributes.

        Args:
             prediction: the prediction (binary) (N, Pts)
             target: the groundtruth (binary) (N, Pts)
        """
        tp = int(np.sum(np.logical_and(prediction == 1, target == 1)))
        fp = int(np.sum(np.logical_and(prediction == 1, target == 0)))
        tn = int(np.sum(np.logical_and(prediction == 0, target == 0)))
        fn = int(np.sum(np.logical_and(prediction == 0, target == 1)))
        self.true_positives.append(tp)
        self.false_positives.append(fp)
        self.true_negatives.append(tn)
        self.false_negatives.append(fn)
        self.n_samples.append(tp + fp + tn + fn)

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return [self.METRIC_BINARY]

    def reset(self) -> None:
        """Reset the saved predictions to start new round of evaluation."""
        self.true_positives = []
        self.false_positives = []
        self.true_negatives = []
        self.false_negatives = []
        self.n_samples = []

    def process_batch(
        self,
        prediction: ArrayLike,
        groundtruth: ArrayLike,
    ) -> None:
        """Processes a new (batch) of predictions.

        Calculates the metrics and caches them internally.

        Args:
             prediction: the prediction(continuous values or bin) (Batch x Pts)
             groundtruth: the groundtruth (binary) (Batch x Pts)
        """
        pred, gt = threshold_and_flatten(
            array_to_numpy(prediction, n_dims=None, dtype=np.float32),
            array_to_numpy(groundtruth, n_dims=None, dtype=np.bool_),
            self.threshold,
        )

        # Confusion Matrix
        self._calc_confusion_matrix(pred, gt)
        self.has_samples = True

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate predictions.

        Returns a dict containing the raw data and a
        short description string containing a readable result.

        Args:
            metric (str): Metric to use. See @property metric

        Returns:
            metric_data, description
            tuple containing the metric data (dict with metric name and value)
            as well as a short string with shortened information.

        Raises:
            RuntimeError: if no data has been registered to be evaluated.
            ValueError: if metric is not supported.
        """
        if not self.has_samples:
            raise RuntimeError(
                """No data registered to calculate metric.
                   Register data using .process() first!"""
            )
        metric_data: MetricLogs = {}
        short_description = ""

        if metric == self.METRIC_BINARY:
            # IoU
            iou = sum(self.true_positives) / (
                sum(self.n_samples) - sum(self.true_negatives) + 1e-6
            )
            metric_data[self.KEY_IOU] = iou
            short_description += f"IoU: {iou:.3f}\n"

            # Accuracy
            acc = (sum(self.true_positives) + sum(self.true_negatives)) / sum(
                self.n_samples
            )
            metric_data[self.KEY_ACCURACY] = acc
            short_description += f"Accuracy: {acc:.3f}\n"

            # Precision
            tp_fp = sum(self.true_positives) + sum(self.false_positives)
            precision = sum(self.true_positives) / tp_fp if tp_fp != 0 else 1
            metric_data[self.KEY_PRECISION] = precision
            short_description += f"Precision: {precision:.3f}\n"

            # Recall
            tp_fn = sum(self.true_positives) + sum(self.false_negatives)
            recall = sum(self.true_positives) / tp_fn if tp_fn != 0 else 1
            metric_data[self.KEY_RECALL] = recall
            short_description += f"Recall: {acc:.3f}\n"

            # F1
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            metric_data[self.KEY_F1] = f1
            short_description += f"F1: {f1:.3f}\n"

        else:
            raise ValueError(
                f"Unsupported metric: {metric}"
            )  # pragma: no cover

        return metric_data, short_description
