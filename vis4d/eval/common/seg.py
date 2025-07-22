"""Common segmentation evaluator."""

from __future__ import annotations

import numpy as np
from terminaltables import AsciiTable

from vis4d.common import MetricLogs
from vis4d.common.array import array_to_numpy
from vis4d.common.typing import ArrayLike, NDArrayI64, NDArrayNumber
from vis4d.eval.base import Evaluator


class SegEvaluator(Evaluator):
    """Creates an evaluator that calculates mIoU score and confusion matrix."""

    METRIC_MIOU = "mIoU"
    METRIC_CONFUSION_MATRIX = "confusion_matrix"

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

        self._confusion_matrix: NDArrayI64 | None = None
        self.reset()

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return [
            self.METRIC_MIOU,
            self.METRIC_CONFUSION_MATRIX,
        ]

    # Taken and modified (added static N) from
    # https://stackoverflow.com/questions/59080843/faster-method-of-computing-confusion-matrix
    def calc_confusion_matrix(
        self, prediction: NDArrayNumber, groundtruth: NDArrayI64
    ) -> NDArrayI64:
        """Calculates the confusion matrix for multi class predictions.

        Args:
            prediction (array): Class predictions
            groundtruth (array): Groundtruth classes

        Returns:
            Confusion Matrix of dimension n_classes x n_classes.
        """
        y_true = groundtruth.reshape(-1)
        if prediction.shape != groundtruth.shape:
            y_pred = np.argmax(prediction, axis=1).reshape(-1)
        else:
            y_pred = prediction.reshape(-1)
        y_pred = y_pred.astype(np.int64)

        if self.class_to_ignore is not None:
            valid = y_true != self.class_to_ignore
            y_true = y_true[valid]
            y_pred = y_pred[valid]
        if self.num_classes is None:
            n_classes = np.max(np.max(groundtruth), np.max(y_pred)) + 1
        else:
            n_classes = self.num_classes

        y = n_classes * y_true + y_pred
        y = np.bincount(y, minlength=n_classes * n_classes)
        return y.reshape(n_classes, n_classes)

    def reset(self) -> None:
        """Reset the saved predictions to start new round of evaluation."""
        self._confusion_matrix = None

    def process_batch(
        self, prediction: ArrayLike, groundtruth: ArrayLike
    ) -> None:
        """Process sample and update confusion matrix.

        Args:
             prediction: Predictions of shape [N,C,...] or [N,...] with
                    C* being any number if channels. Note, C is passed,
                    the prediction is converted to target labels by applying
                    the max operations along the second axis
             groundtruth: Groundtruth of shape [N_batch, ...] type int
        """
        confusion_matrix = self.calc_confusion_matrix(
            array_to_numpy(prediction, n_dims=None, dtype=np.float32),
            array_to_numpy(groundtruth, n_dims=None, dtype=np.int64),
        )

        if self._confusion_matrix is None:
            self._confusion_matrix = confusion_matrix
        else:
            assert (
                self._confusion_matrix.shape == confusion_matrix.shape
            ), """Shape of confusion matrix changed during runtime!,
                  Please specify a static number of classes in constructor."""
            self._confusion_matrix += confusion_matrix

    def _get_class_name_for_idx(self, idx: int) -> str:
        """Maps a class index to a unique class name.

        Args:
            idx (int): class index.

        Returns:
            (str) class name
        """
        return self.class_mapping.get(idx, f"class_{idx}")

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate predictions.

        Returns a dict containing the raw data and a
        short description string containing a readable result.

        Args:
            metric (str): Metric to use. See @property metric.

        Returns:
            (dict, str) containing the raw data and a short description string.

        Raises:
            ValueError: If metric is not supported.
        """
        assert (
            self._confusion_matrix is not None
        ), """Evaluate() needs to process samples first.
            Please call the process() function before calling evaluate()"""

        metric_data, short_description = {}, ""
        if metric == self.METRIC_MIOU:
            # Calculate miou from confusion matrix
            tp = np.diag(self._confusion_matrix)
            fp = np.sum(self._confusion_matrix, axis=0) - tp
            fn = np.sum(self._confusion_matrix, axis=1) - tp
            iou = tp / (tp + fn + fp) * 100
            m_iou = np.nanmean(iou)

            iou_class_str = ", ".join(
                f"{self._get_class_name_for_idx(idx)}: ({d:.3f}%)"
                for idx, d in enumerate(iou)
            )
            metric_data[self.METRIC_MIOU] = m_iou
            short_description += f"mIoU: {m_iou:.3f}% \n"
            short_description += iou_class_str + "\n"

        elif metric == self.METRIC_CONFUSION_MATRIX:
            headers = ["Confusion"] + [
                self._get_class_name_for_idx(i)
                for i in range(self._confusion_matrix.shape[0])
            ]
            table_data = self._confusion_matrix / (
                np.sum(self._confusion_matrix, axis=1)
            )
            data = list(
                [f"Class_{idx}"] + list(d) for idx, d in enumerate(table_data)
            )
            table = AsciiTable([headers] + data)
            # TODO, change MetricLogs type for more complex log types as e.g.
            #       confusion matrix
            short_description += table.table + "\n"

        else:
            raise ValueError(f"Metric {metric} not supported")
        return metric_data, short_description
