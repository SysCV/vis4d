"""Accuracy Evaluator."""
from __future__ import annotations

import numpy as np

from vis4d.common import MetricLogs
from vis4d.common.typing import NDArrayI64, NDArrayNumber
from vis4d.eval.base import Evaluator


class ClassificationEvaluator(Evaluator):
    """Creates an evaluator that calculates accuracy for classification."""

    METRIC_ACCURACY = "accuracy"

    def __init__(
        self,
        num_classes: int | None = None,
        class_to_ignore: int | None = None,
        class_mapping: dict[int, str] | None = None,
    ):
        """Creates a new evaluator.

        Args:
            num_classes (int): Number of semantic classes
            class_to_ignore (int | None): Groundtruth class that should be
                                            ignored
            class_mapping (int): dict mapping each class_id to a readable name

        """
        super().__init__()
        self.num_classes = num_classes
        self.class_mapping = class_mapping if class_mapping is not None else {}
        self.class_to_ignore = class_to_ignore

        self.reset()

    @property
    def metrics(self) -> list[str]:
        """Return list of metrics to evaluate.

        Returns:
            list[str]: Metrics to evaluate.

        """
        return [self.METRIC_ACCURACY]

    def reset(self) -> None:
        """Reset the evaluator."""
        self.top5_pred_classes = []
        self.labels = []

    def process(  # type: ignore # pylint: disable=arguments-differ
        self, predictions: NDArrayNumber, labels: NDArrayI64
    ) -> None:
        """Process predictions and labels.

        Args:
            predictions (NDArrayNumber): Predictions, shape (N, num_classes).
            labels (NDArrayI64): Labels, shape (N,).
        """
        self.top5_pred_classes.append(np.argsort(predictions, axis=1)[:, -5:])
        self.labels.append(labels)

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate the metric.

        Args:
            metric (str): Metric to evaluate.

        Returns:
            tuple[MetricLogs, str]: Metric logs and metric name.
        """
        assert len(self.labels) == len(self.top5_pred_classes), (
            f"Number of predictions {len(self.top5_pred_classes)} "
            f"does not match number of labels {len(self.labels)}."
        )
        assert len(self.labels) > 0, (
            "Evaluate() needs to process samples first. Please call the "
            "process() function before calling evaluate()."
        )

        metric_data, short_description = {}, ""
        top5_pred_classes = np.concatenate(self.top5_pred_classes, axis=0)
        pred_classes = np.argmax(top5_pred_classes, axis=1)
        labels = np.concatenate(self.labels, axis=0)

        if metric == self.METRIC_ACCURACY:
            accuracy = np.mean(pred_classes == labels)
            top5_accuracy = np.mean(
                np.any(top5_pred_classes == labels[:, np.newaxis], axis=1)
            )
            metric_data["accuracy"] = accuracy
            metric_data["top1_error"] = 1.0 - accuracy
            metric_data["top5_error"] = 1.0 - top5_accuracy
            short_description = (
                f"Accuracy@Top1: {accuracy:.4f}, "
                f"Accuracy@Top5: {top5_accuracy:.4f}, "
                f"Error@Top1: {1.0 - accuracy:.4f}, "
                f"Error@Top5: {1.0 - top5_accuracy:.4f}"
            )
        else:
            raise ValueError(f"Metric {metric} is not supported.")
        return metric_data, short_description
