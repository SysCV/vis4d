"""Depth eval test cases."""
from __future__ import annotations

import unittest

import numpy as np

from vis4d.common.typing import NDArrayNumber
from vis4d.eval.common import OpticalFlowEvaluator


def get_test_metrics() -> tuple[NDArrayNumber, NDArrayNumber]:
    """Precomputed input metrics."""
    pred = np.asarray(
        [
            [[1, 3], [3, 2], [0, -1], [2, 4]],
            [[2, 1], [1, 3], [3, 2], [0, -1]],
            [[3, 2], [0, -1], [2, 4], [-2, 1]],
            [[1, 3], [3, 2], [0, -1], [2, 4]],
        ],
        dtype=np.float32,
    )

    gt = np.asarray(
        [
            [[4, 0], [2, 1], [1, 3], [3, 2]],
            [[3, 2], [0, -1], [2, 4], [-2, 1]],
            [[-2, 5], [0, 0], [-3, -1], [2, -4]],
            [[3, -4], [1, 2], [0, 0], [2, 1]],
        ],
        dtype=np.float32,
    )
    return pred[np.newaxis, ...], gt[np.newaxis, ...]


class TestOpticalFlowEvaluator(unittest.TestCase):
    """Tests for OpticalFlowEvaluator."""

    batch_size = 4
    mask_h, mask_w = 10, 10
    flow_scale = 40.0
    evaluator = OpticalFlowEvaluator()

    def test_perfect_prediction(self) -> None:
        """Tests when predictions are correct."""
        pred = (
            np.random.rand(self.batch_size, self.mask_h, self.mask_w, 2)
            * self.flow_scale
        )
        gt = pred.copy()
        self.evaluator.reset()
        self.evaluator.process_batch(pred, gt)
        metrics, _ = self.evaluator.evaluate(OpticalFlowEvaluator.METRIC_ALL)
        self.assertAlmostEqual(
            metrics[OpticalFlowEvaluator.METRIC_ENDPOINT_ERROR], 0.0, places=3
        )
        self.assertAlmostEqual(
            metrics[OpticalFlowEvaluator.METRIC_ANGULAR_ERROR], 0.0, places=3
        )

    def test_precomputed(self) -> None:
        """Numerical tests with precomputed values."""
        # All ones
        pred, gt = get_test_metrics()
        self.evaluator.reset()
        self.evaluator.process_batch(pred, gt)
        metrics, log_str = self.evaluator.evaluate(
            OpticalFlowEvaluator.METRIC_ALL
        )
        self.assertAlmostEqual(
            metrics[OpticalFlowEvaluator.METRIC_ENDPOINT_ERROR],
            3.513,
            places=3,
        )
        self.assertAlmostEqual(
            metrics[OpticalFlowEvaluator.METRIC_ANGULAR_ERROR], 0.772, places=3
        )
        assert isinstance(log_str, str)
