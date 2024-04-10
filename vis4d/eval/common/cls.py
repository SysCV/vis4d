"""Image classification evaluator."""

from __future__ import annotations

import itertools

import numpy as np

from vis4d.common.array import array_to_numpy
from vis4d.common.typing import (
    ArrayLike,
    GenericFunc,
    MetricLogs,
    NDArrayI64,
    NDArrayNumber,
)
from vis4d.eval.base import Evaluator

from ..metrics.cls import accuracy


class ClassificationEvaluator(Evaluator):
    """Multi-class classification evaluator."""

    METRIC_CLASSIFICATION = "Cls"

    KEY_ACCURACY = "Acc@1"
    KEY_ACCURACY_TOP5 = "Acc@5"

    def __init__(self) -> None:
        """Initialize the classification evaluator."""
        super().__init__()
        self._metrics_list: list[dict[str, float]] = []

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return [
            self.KEY_ACCURACY,
            self.KEY_ACCURACY_TOP5,
        ]

    def reset(self) -> None:
        """Reset evaluator for new round of evaluation."""
        self._metrics_list = []

    def _is_correct(
        self, pred: NDArrayNumber, target: NDArrayI64, top_k: int = 1
    ) -> bool:
        """Check if the prediction is correct for top-k.

        Args:
            pred (NDArrayNumber): Prediction logits, in shape (C, ).
            target (NDArrayI64): Target logits, in shape (1, ).
            top_k (int, optional): Top-k to check. Defaults to 1.

        Returns:
            bool: Whether the prediction is correct.
        """
        top_k = min(top_k, pred.shape[0])
        top_k_idx = np.argsort(pred)[-top_k:]
        return bool(np.any(top_k_idx == target))

    def process_batch(  # type: ignore # pylint: disable=arguments-differ
        self, prediction: ArrayLike, groundtruth: ArrayLike
    ):
        """Process a batch of predictions and groundtruths.

        Args:
            prediction (ArrayLike): Prediction, in shape (N, C).
            groundtruth (ArrayLike): Groundtruth, in shape (N, ).
        """
        pred = array_to_numpy(prediction, n_dims=None, dtype=np.float32)
        gt = array_to_numpy(groundtruth, n_dims=None, dtype=np.int64)
        for i in range(pred.shape[0]):
            self._metrics_list.append(
                {
                    "top1_correct": accuracy(pred[i], gt[i], top_k=1),
                    "top5_correct": accuracy(pred[i], gt[i], top_k=5),
                }
            )

    def gather(self, gather_func: GenericFunc) -> None:
        """Accumulate predictions across processes."""
        all_metrics = gather_func(self._metrics_list)
        if all_metrics is not None:
            self._metrics_list = list(itertools.chain(*all_metrics))

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
            ValueError: if the metric is not supported.
        """
        if len(self._metrics_list) == 0:
            raise RuntimeError(
                """No data registered to calculate metric.
                   Register data using .process() first!"""
            )
        metric_data: MetricLogs = {}
        short_description = ""

        if metric == self.METRIC_CLASSIFICATION:
            # Top1 accuracy
            top1_correct = np.array(
                [metric["top1_correct"] for metric in self._metrics_list]
            )
            top1_acc = np.mean(top1_correct)
            metric_data[self.KEY_ACCURACY] = top1_acc
            short_description += f"Top1 Accuracy: {top1_acc:.4f}\n"

            # Top5 accuracy
            top5_correct = np.array(
                [metric["top5_correct"] for metric in self._metrics_list]
            )
            top5_acc = np.mean(top5_correct)
            metric_data[self.KEY_ACCURACY_TOP5] = top5_acc
            short_description += f"Top5 Accuracy: {top5_acc:.4f}\n"

        else:
            raise ValueError(
                f"Unsupported metric: {metric}"
            )  # pragma: no cover

        return metric_data, short_description
