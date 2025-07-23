"""Optical flow evaluator."""

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

from ..metrics.flow import angular_error, end_point_error


class OpticalFlowEvaluator(Evaluator):
    """Optical flow evaluator."""

    METRIC_FLOW = "Flow"

    KEY_ENDPOINT_ERROR = "EPE"
    KEY_ANGULAR_ERROR = "AE"

    def __init__(
        self,
        max_flow: float = 400.0,
        use_degrees: bool = False,
        scale: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        """Initialize the optical flow evaluator.

        Args:
            max_flow (float, optional): Maximum flow value. Defaults to 400.0.
            use_degrees (bool, optional): Whether to use degrees for angular
                error. Defaults to False.
            scale (float, optional): Scale factor for the optical flow.
                Defaults to 1.0.
            epsilon (float, optional): Epsilon value for numerical stability.
        """
        super().__init__()
        self.max_flow = max_flow
        self.use_degrees = use_degrees
        self.scale = scale
        self.epsilon = epsilon
        self._metrics_list: list[dict[str, float]] = []

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return [
            OpticalFlowEvaluator.METRIC_FLOW,
        ]

    def reset(self) -> None:
        """Reset evaluator for new round of evaluation."""
        self._metrics_list = []

    def _apply_mask(
        self, prediction: NDArrayFloat, target: NDArrayFloat
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Apply mask to prediction and target."""
        mask = np.sum(np.abs(target), axis=-1) <= self.max_flow
        return prediction[mask], target[mask]

    def process_batch(
        self, prediction: ArrayLike, groundtruth: ArrayLike
    ) -> None:
        """Process a batch of data.

        Args:
            prediction (NDArrayNumber): Prediction optical flow, in shape
                (N, H, W, 2).
            groundtruth (NDArrayNumber): Target optical flow, in shape
                (N, H, W, 2).
        """
        preds = (
            array_to_numpy(prediction, n_dims=None, dtype=np.float32)
            * self.scale
        )
        gts = array_to_numpy(groundtruth, n_dims=None, dtype=np.float32)

        for pred, gt in zip(preds, gts):
            pred, gt = self._apply_mask(pred, gt)
            epe = end_point_error(pred, gt)
            ae = angular_error(pred, gt, self.epsilon)
            self._metrics_list.append(
                {
                    OpticalFlowEvaluator.KEY_ENDPOINT_ERROR: epe,
                    OpticalFlowEvaluator.KEY_ANGULAR_ERROR: ae,
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
            ValueError: if metric is not supported.
        """
        if len(self._metrics_list) == 0:
            raise RuntimeError(
                """No data registered to calculate metric.
                   Register data using .process() first!"""
            )
        metric_data: MetricLogs = {}
        short_description = ""

        if metric == OpticalFlowEvaluator.METRIC_FLOW:
            # EPE
            epe = np.mean(
                [x[self.KEY_ENDPOINT_ERROR] for x in self._metrics_list]
            )
            metric_data[self.KEY_ENDPOINT_ERROR] = float(epe)
            short_description = f"EPE: {epe:.3f}"

            # AE
            ae = np.mean(
                [x[self.KEY_ANGULAR_ERROR] for x in self._metrics_list]
            )
            metric_data[self.KEY_ANGULAR_ERROR] = float(ae)
            angular_unit = "rad" if not self.use_degrees else "deg"
            short_description = f"AE: {ae:.3f}{angular_unit}"

        else:
            raise ValueError(
                f"Unsupported metric: {metric}"
            )  # pragma: no cover

        return metric_data, short_description
