"""Accuracy Evaluator."""
from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

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
        self._records = []

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
        self._records = []

    def gather(  # type: ignore
        self, gather_func: Callable[[Any], Any]
    ) -> None:
        """Accumulate predictions across processes."""
        all_preds = gather_func(self._records)
        if all_preds is not None:
            self._records = list(itertools.chain(*all_preds))

    def process(  # type: ignore # pylint: disable=arguments-differ
        self, predictions: NDArrayNumber, labels: NDArrayI64
    ) -> None:
        """Process predictions and labels.

        Args:
            predictions (NDArrayNumber): Predictions, shape (N, num_classes).
            labels (NDArrayI64): Labels, shape (N,).
        """
        self._records.append((np.argsort(predictions, axis=1)[:, -5:], labels))

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate the metric.

        Args:
            metric (str): Metric to evaluate.

        Returns:
            tuple[MetricLogs, str]: Metric logs and metric name.
        """
        assert len(self._records) > 0, (
            "Evaluate() needs to process samples first. Please call the "
            "process() function before calling evaluate()."
        )

        metric_data, short_description = {}, ""
        top5_pred_classes, labels = zip(*self._records)
        top5_pred_classes = np.concatenate(top5_pred_classes, axis=0)
        labels = np.concatenate(labels, axis=0)

        if metric == self.METRIC_ACCURACY:
            accuracy = np.mean(top5_pred_classes[:, -1] == labels)
            top5_accuracy = np.mean(
                np.any(top5_pred_classes == labels[:, np.newaxis], axis=1)
            )
            metric_data["top1_accuracy"] = accuracy
            metric_data["top5_accuracy"] = top5_accuracy
            metric_data["top1_error"] = 1.0 - accuracy
            metric_data["top5_error"] = 1.0 - top5_accuracy
            short_description = (
                f"Acc@1: {accuracy:.4f}, "
                f"Acc@5: {top5_accuracy:.4f}, "
                f"Err@1: {1.0 - accuracy:.4f}, "
                f"Err@5: {1.0 - top5_accuracy:.4f}, "
                f"Num: {len(labels)}"
            )
        else:
            raise ValueError(f"Metric {metric} is not supported.")
        return metric_data, short_description
