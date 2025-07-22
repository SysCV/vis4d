"""Depth estimation evaluator."""

from __future__ import annotations

import itertools

import numpy as np

from vis4d.common.array import array_to_numpy
from vis4d.common.typing import (
    ArrayLike,
    GenericFunc,
    MetricLogs,
    NDArrayFloat,
)
from vis4d.eval.base import Evaluator

from ..metrics.depth import (
    absolute_error,
    absolute_relative_error,
    delta_p,
    log_10_error,
    root_mean_squared_error,
    root_mean_squared_error_log,
    scale_invariant_log,
    squared_relative_error,
)


class DepthEvaluator(Evaluator):
    """Depth estimation evaluator."""

    METRIC_DEPTH = "Depth"

    KEY_DELTA05 = "d05"
    KEY_DELTA1 = "d1"
    KEY_DELTA2 = "d2"
    KEY_DELTA3 = "d3"

    KEY_ABS_REL = "AbsRel"
    KEY_ABS_ERR = "AbsErr"
    KEY_SQ_REL = "SqRel"
    KEY_RMSE = "RMSE"
    KEY_RMSE_LOG = "RMSELog"
    KEY_SILOG = "SILog"
    KEY_LOG10 = "Log10"

    def __init__(
        self,
        min_depth: float = 0.0,
        max_depth: float = 80.0,
        scale: float = 1.0,
        epsilon: float = 1e-3,
    ) -> None:
        """Initialize the optical flow evaluator.

        Args:
            min_depth (float): Minimum depth to evaluate. Defaults to 0.001.
            max_depth (float): Maximum depth to evaluate. Defaults to 80.0.
            scale (float): Scale factor for depth. Defaults to 1.0.
            epsilon (float): Small value to avoid logarithms of small values.
                Defaults to 1e-3.
        """
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.epsilon = epsilon
        self.scale = scale
        self._metrics_list: list[dict[str, float]] = []

    def __repr__(self) -> str:
        """Concise representation of the evaluator."""
        return "Common Depth Evaluator"

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return [self.METRIC_DEPTH]

    def reset(self) -> None:
        """Reset evaluator for new round of evaluation."""
        self._metrics_list = []

    def gather(self, gather_func: GenericFunc) -> None:
        """Accumulate predictions across processes."""
        all_metrics = gather_func(self._metrics_list)
        if all_metrics is not None:
            self._metrics_list = list(itertools.chain(*all_metrics))

    def _apply_mask(
        self, prediction: NDArrayFloat, target: NDArrayFloat
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Apply mask to prediction and target."""
        mask = (target > self.min_depth) & (target <= self.max_depth)
        return prediction[mask], target[mask]

    def process_batch(
        self, prediction: ArrayLike, groundtruth: ArrayLike
    ) -> None:
        """Process a batch of data.

        Args:
            prediction (np.array): Prediction optical flow, in shape (B, H, W).
            groundtruth (np.array): Target optical flow, in shape (B, H, W).
        """
        preds = (
            array_to_numpy(prediction, n_dims=None, dtype=np.float32)
            * self.scale
        )
        gts = array_to_numpy(groundtruth, n_dims=None, dtype=np.float32)

        for pred, gt in zip(preds, gts):
            pred, gt = self._apply_mask(pred, gt)
            self._metrics_list.append(
                {
                    self.KEY_ABS_REL: absolute_relative_error(pred, gt),
                    self.KEY_ABS_ERR: absolute_error(pred, gt),
                    self.KEY_SQ_REL: squared_relative_error(pred, gt),
                    self.KEY_RMSE: root_mean_squared_error(pred, gt),
                    self.KEY_RMSE_LOG: root_mean_squared_error_log(pred, gt),
                    self.KEY_SILOG: scale_invariant_log(pred, gt),
                    self.KEY_DELTA05: delta_p(pred, gt, 0.5),
                    self.KEY_DELTA1: delta_p(pred, gt, 1.0),
                    self.KEY_DELTA2: delta_p(pred, gt, 2.0),
                    self.KEY_DELTA3: delta_p(pred, gt, 3.0),
                    self.KEY_LOG10: log_10_error(pred, gt),
                }
            )

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate predictions.

        Returns a dict containing the raw data and a
        short description string containing a readablae result.

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
        if len(self._metrics_list) == 0:
            raise RuntimeError(
                """No data registered to calculate metric.
                   Register data using .process() first!"""
            )
        metric_data: MetricLogs = {}
        short_description = "\n"

        if metric == self.METRIC_DEPTH:
            abs_rel = np.mean(
                [x[self.KEY_ABS_REL] for x in self._metrics_list]
            )
            metric_data[self.KEY_ABS_REL] = float(abs_rel)
            short_description += f"Absolute relative error: {abs_rel:.3f}\n"

            abs_err = np.mean(
                [x[self.KEY_ABS_ERR] for x in self._metrics_list]
            )
            metric_data[self.KEY_ABS_ERR] = float(abs_err)
            short_description += f"Absolute error: {abs_err:.3f}\n"

            sq_rel = np.mean([x[self.KEY_SQ_REL] for x in self._metrics_list])
            metric_data[self.KEY_SQ_REL] = float(sq_rel)
            short_description += f"Squared relative error: {sq_rel:.3f}\n"

            rmse = np.mean([x[self.KEY_RMSE] for x in self._metrics_list])
            metric_data[self.KEY_RMSE] = float(rmse)
            short_description += f"RMSE: {rmse:.3f}\n"

            rmse_log = np.mean(
                [x[self.KEY_RMSE_LOG] for x in self._metrics_list]
            )
            metric_data[self.KEY_RMSE_LOG] = float(rmse_log)
            short_description += f"RMSE log: {rmse_log:.3f}\n"

            silog = np.mean([x[self.KEY_SILOG] for x in self._metrics_list])
            metric_data[self.KEY_SILOG] = float(silog)
            short_description += f"SILog: {silog:.3f}\n"

            delta05 = np.mean(
                [x[self.KEY_DELTA05] for x in self._metrics_list]
            )
            metric_data[self.KEY_DELTA05] = float(delta05)
            short_description += f"Delta 0.5: {delta05:.3f}\n"

            delta1 = np.mean([x[self.KEY_DELTA1] for x in self._metrics_list])
            metric_data[self.KEY_DELTA1] = float(delta1)
            short_description += f"Delta 1: {delta1:.3f}\n"

            delta2 = np.mean([x[self.KEY_DELTA2] for x in self._metrics_list])
            metric_data[self.KEY_DELTA2] = float(delta2)
            short_description += f"Delta 2: {delta2:.3f}\n"

            delta3 = np.mean([x[self.KEY_DELTA3] for x in self._metrics_list])
            metric_data[self.KEY_DELTA3] = float(delta3)
            short_description += f"Delta 3: {delta3:.3f}\n"

            log10 = np.mean([x[self.KEY_LOG10] for x in self._metrics_list])
            metric_data[self.KEY_LOG10] = float(log10)
            short_description += f"Log10 error: {log10:.3f}\n"

        else:
            raise ValueError(
                f"Unsupported metric: {metric}"
            )  # pragma: no cover

        return metric_data, short_description
