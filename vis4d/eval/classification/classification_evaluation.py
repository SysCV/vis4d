"""Accuracy Evaluator."""
from __future__ import annotations

import numpy as np
from terminaltables import AsciiTable  # type: ignore

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

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate the metric.

        Args:
            metric (str): Metric to evaluate.

        Returns:
            tuple[MetricLogs, str]: Metric logs and metric name.

        """
        if metric == ClassificationEvaluator.METRIC_ACCURACY:
            return self.accuracy(), ClassificationEvaluator.METRIC_ACCURACY
        else:
            raise ValueError(f"Metric {metric} is not supported.")