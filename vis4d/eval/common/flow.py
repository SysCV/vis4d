"""Optical flow evaluator."""

from __future__ import annotations

import itertools

import numpy as np

from vis4d.common.typing import GenericFunc, MetricLogs, NDArrayNumber
from vis4d.eval.base import Evaluator


class OpticalFlowEvaluator(Evaluator):
    """Optical flow evaluator."""

    METRIC_ENDPOINT_ERROR = "EPE"
    METRIC_ANGULAR_ERROR = "AE"
    METRIC_ALL = "all"

    def __init__(
        self,
        max_flow: float = 400.0,
        use_degrees: bool = False,
        scale: float = 1.0,
    ) -> None:
        """Initialize the optical flow evaluator.

        Args:
            max_flow (float, optional): Maximum flow value. Defaults to 400.0.
            use_degrees (bool, optional): Whether to use degrees for angular
                error. Defaults to False.
            scale (float, optional): Scale factor for the optical flow.
                Defaults to 1.0.
        """
        super().__init__()
        self.max_flow = max_flow
        self.use_degrees = use_degrees
        self.scale = scale
        self._metrics_list: list[dict[str, float]] = []

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return [
            OpticalFlowEvaluator.METRIC_ENDPOINT_ERROR,
            OpticalFlowEvaluator.METRIC_ANGULAR_ERROR,
        ]

    def reset(self) -> None:
        """Reset evaluator for new round of evaluation."""
        self._metrics_list = []

    def end_point_error(
        self, prediction: NDArrayNumber, target: NDArrayNumber
    ) -> float:
        """Compute the end point error.

        Args:
            prediction (np.array): Prediction optical flow, in shape (H, W, 2).
            target (np.array): Target optical flow, in shape (H, W, 2).

        Returns:
            float: End point error.
        """
        mask = np.sum(np.abs(target), axis=2) < self.max_flow
        return np.mean(
            np.sqrt(np.sum((prediction[mask] - target[mask]) ** 2, axis=1))
        )

    def angular_error(
        self, prediction: NDArrayNumber, target: NDArrayNumber
    ) -> float:
        """Compute the angular error.

        Args:
            prediction (np.array): Prediction optical flow, in shape (H, W, 2).
            target (np.array): Target optical flow, in shape (H, W, 2).

        Returns:
            float: Angular error.
        """
        mask = np.sum(np.abs(target), axis=2) < self.max_flow
        return np.mean(
            np.arccos(
                np.sum(prediction[mask] * target[mask], axis=1)
                / (
                    np.linalg.norm(prediction[mask], axis=1)
                    * np.linalg.norm(target[mask], axis=1)
                )
            )
        )

    def process(  # type: ignore # pylint: disable=arguments-differ
        self, prediction: NDArrayNumber, groundtruth: NDArrayNumber
    ) -> None:
        """Process a batch of data.

        Args:
            prediction (NDArrayNumber): Prediction optical flow, in shape
                (N, H, W, 2).
            groundtruth (NDArrayNumber): Target optical flow, in shape
                (N, H, W, 2).
        """
        pred_scaled = prediction * self.scale
        for i in range(pred_scaled.shape[0]):
            epe = self.end_point_error(pred_scaled[i], groundtruth[i])
            ae = self.angular_error(pred_scaled[i], groundtruth[i])
            self._metrics_list.append(
                {
                    OpticalFlowEvaluator.METRIC_ENDPOINT_ERROR: epe,
                    OpticalFlowEvaluator.METRIC_ANGULAR_ERROR: ae,
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
            OpticalFlowEvaluator.METRIC_ENDPOINT_ERROR,
            OpticalFlowEvaluator.METRIC_ALL,
        ]:
            metric_data[self.METRIC_ENDPOINT_ERROR] = np.mean(
                [x[self.METRIC_ENDPOINT_ERROR] for x in self._metrics_list]
            )
            short_description = f"EPE: {metric_data[metric]:.3f}"
        elif metric in [
            OpticalFlowEvaluator.METRIC_ANGULAR_ERROR,
            OpticalFlowEvaluator.METRIC_ALL,
        ]:
            metric_data[self.METRIC_ANGULAR_ERROR] = np.mean(
                [x[self.METRIC_ANGULAR_ERROR] for x in self._metrics_list]
            )
            angular_unit = "rad" if not self.use_degrees else "deg"
            short_description = f"AE: {metric_data[metric]:.3f}{angular_unit}"

        return metric_data, short_description
