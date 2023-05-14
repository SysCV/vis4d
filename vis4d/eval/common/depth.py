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
    root_mean_squared_error,
    root_mean_squared_error_log,
    scale_invariant_log,
    squared_relative_error,
)


class DepthEvaluator(Evaluator):
    """Depth estimation evaluator."""

    METRIC_ABS_REL = "AbsRel"
    METRIC_ABS_ERR = "AbsErr"
    METRIC_SQ_REL = "SqRel"
    METRIC_RMSE = "RMSE"
    METRIC_RMSE_LOG = "RMSELog"
    METRIC_SILOG = "SILog"
    METRIC_ALL = "all"

    def __init__(
        self,
        min_depth: float = 0.001,
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

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return [
            DepthEvaluator.METRIC_ABS_REL,
            DepthEvaluator.METRIC_ABS_ERR,
            DepthEvaluator.METRIC_SQ_REL,
            DepthEvaluator.METRIC_RMSE,
            DepthEvaluator.METRIC_RMSE_LOG,
            DepthEvaluator.METRIC_SILOG,
        ]

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
        mask = (target > self.min_depth) & (target < self.max_depth)
        return prediction[mask], target[mask]

    def process_batch(  # type: ignore # pylint: disable=arguments-differ
        self, prediction: ArrayLike, groundtruth: ArrayLike
    ) -> None:
        """Process a batch of data.

        Args:
            prediction (np.array): Prediction optical flow, in shape (H, W, 2).
            groundtruth (np.array): Target optical flow, in shape (H, W, 2).
        """
        preds = (
            array_to_numpy(prediction, n_dims=None, dtype=np.float32)
            * self.scale
        )
        gts = array_to_numpy(groundtruth, n_dims=None, dtype=np.float32)
        preds, gts = self._apply_mask(preds, gts)

        for pred, gt in zip(preds, gts):
            self._metrics_list.append(
                {
                    self.METRIC_ABS_REL: absolute_relative_error(pred, gt),
                    self.METRIC_ABS_ERR: absolute_error(pred, gt),
                    self.METRIC_SQ_REL: squared_relative_error(pred, gt),
                    self.METRIC_RMSE: root_mean_squared_error(pred, gt),
                    self.METRIC_RMSE_LOG: root_mean_squared_error_log(
                        pred, gt
                    ),
                    self.METRIC_SILOG: scale_invariant_log(pred, gt),
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
            RuntimeError: if no data has been registered to be evaluated
        """
        if len(self._metrics_list) == 0:
            raise RuntimeError(
                """No data registered to calculate metric.
                   Register data using .process() first!"""
            )
        metric_data: MetricLogs = {}
        short_description = ""

        if metric in [
            DepthEvaluator.METRIC_ABS_REL,
            DepthEvaluator.METRIC_ALL,
        ]:
            abs_rel = np.mean(
                [x[self.METRIC_ABS_REL] for x in self._metrics_list]
            )
            metric_data[self.METRIC_ABS_REL] = abs_rel
            short_description += f"Absolute relative error: {abs_rel:.3f} "
        if metric in [
            DepthEvaluator.METRIC_ABS_ERR,
            DepthEvaluator.METRIC_ALL,
        ]:
            abs_err = np.mean(
                [x[self.METRIC_ABS_ERR] for x in self._metrics_list]
            )
            metric_data[self.METRIC_ABS_ERR] = abs_err
            short_description += f"Absolute error: {abs_err:.3f}\n"
        if metric in [DepthEvaluator.METRIC_SQ_REL, DepthEvaluator.METRIC_ALL]:
            sq_rel = np.mean(
                [x[self.METRIC_SQ_REL] for x in self._metrics_list]
            )
            metric_data[self.METRIC_SQ_REL] = sq_rel
            short_description += f"Squared relative error: {sq_rel:.3f}\n"
        if metric in [DepthEvaluator.METRIC_RMSE, DepthEvaluator.METRIC_ALL]:
            rmse = np.mean([x[self.METRIC_RMSE] for x in self._metrics_list])
            metric_data[self.METRIC_RMSE] = rmse
            short_description += f"RMSE: {rmse:.3f}\n"
        if metric in [
            DepthEvaluator.METRIC_RMSE_LOG,
            DepthEvaluator.METRIC_ALL,
        ]:
            rmse_log = np.mean(
                [x[self.METRIC_RMSE_LOG] for x in self._metrics_list]
            )
            metric_data[self.METRIC_RMSE_LOG] = rmse_log
            short_description += f"RMSE log: {rmse_log:.3f}\n"
        if metric in [DepthEvaluator.METRIC_SILOG, DepthEvaluator.METRIC_ALL]:
            silog = np.mean([x[self.METRIC_SILOG] for x in self._metrics_list])
            metric_data[self.METRIC_SILOG] = silog
            short_description += f"SILog: {silog:.3f}\n"

        return metric_data, short_description
