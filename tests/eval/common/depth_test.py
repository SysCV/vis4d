"""Depth eval test cases."""
from __future__ import annotations

import unittest

import numpy as np

from vis4d.common.typing import NDArrayNumber
from vis4d.eval.common import DepthEvaluator


def get_test_metrics() -> tuple[NDArrayNumber, NDArrayNumber]:
    """Precomputed input metrics."""
    pred = np.asarray(
        [
            [8, 3, 5, 4],
            [8, 3, 2, 1],
            [3, 1, 2, 4],
            [1, 8, 9, 1],
        ],
        dtype=np.float32,
    )

    gt = np.asarray(
        [
            [1, 4, 4, 9],
            [5, 1, 8, 4],
            [1, 5, 8, 5],
            [3, 3, 7, 5],
        ],
        dtype=np.float32,
    )
    return pred, gt


class TestDepthEvaluator(unittest.TestCase):
    """Tests for SegEvaluator."""

    batch_size = 4
    mask_h, mask_w = 10, 10
    depth_scale = 10.0
    evaluator = DepthEvaluator(min_depth=0.0, max_depth=10.0)

    def test_perfect_prediction(self) -> None:
        """Tests when predictions are correct."""
        pred = (
            np.random.rand(self.batch_size, self.mask_h, self.mask_w)
            * self.depth_scale
        )
        gt = pred.copy()
        self.evaluator.reset()
        self.evaluator.process_batch(pred, gt)
        metrics, _ = self.evaluator.evaluate(DepthEvaluator.METRIC_ALL)
        self.assertAlmostEqual(
            metrics[DepthEvaluator.METRIC_RMSE], 0.0, places=3
        )
        self.assertAlmostEqual(
            metrics[DepthEvaluator.METRIC_RMSE_LOG], 0.0, places=3
        )
        self.assertAlmostEqual(
            metrics[DepthEvaluator.METRIC_ABS_ERR], 0.0, places=3
        )
        self.assertAlmostEqual(
            metrics[DepthEvaluator.METRIC_ABS_REL], 0.0, places=3
        )
        self.assertAlmostEqual(
            metrics[DepthEvaluator.METRIC_SQ_REL], 0.0, places=3
        )

    def test_precomputed(self) -> None:
        """Numerical tests with precomputed values."""
        # All ones
        pred, gt = get_test_metrics()
        self.evaluator.reset()
        self.evaluator.process_batch(pred, gt)
        metrics, log_str = self.evaluator.evaluate(DepthEvaluator.METRIC_ALL)
        self.assertAlmostEqual(
            metrics[DepthEvaluator.METRIC_SILOG], 1.801, places=3
        )
        self.assertAlmostEqual(
            metrics[DepthEvaluator.METRIC_RMSE], 3.375, places=3
        )
        self.assertAlmostEqual(
            metrics[DepthEvaluator.METRIC_RMSE_LOG], 1.00, places=3
        )
        self.assertAlmostEqual(
            metrics[DepthEvaluator.METRIC_ABS_ERR], 3.375, places=3
        )
        self.assertAlmostEqual(
            metrics[DepthEvaluator.METRIC_ABS_REL], 1.208, places=3
        )
        self.assertAlmostEqual(
            metrics[DepthEvaluator.METRIC_SQ_REL], 4.007, places=3
        )
        assert isinstance(log_str, str)
