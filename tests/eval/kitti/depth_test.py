"""KITTI Depth Evaluation test."""

from __future__ import annotations

import unittest

import numpy as np

from vis4d.eval.kitti import KITTIDepthEvaluator

from ..common.depth_test import get_test_metrics


class TestKITTIDepthEvaluator(unittest.TestCase):
    """Tests for KITTIDepthEvaluator."""

    batch_size = 4
    mask_h, mask_w = 10, 10
    depth_scale = 10.0
    evaluator = KITTIDepthEvaluator(min_depth=0.0, max_depth=10.0)

    def test_perfect_prediction(self) -> None:
        """Tests when predictions are correct."""
        pred = (
            np.random.rand(self.batch_size, self.mask_h, self.mask_w)
            * self.depth_scale
        )
        gt = pred.copy()
        self.evaluator.reset()
        self.evaluator.process_batch(pred, gt)
        metrics, _ = self.evaluator.evaluate(KITTIDepthEvaluator.METRIC_DEPTH)
        self.assertAlmostEqual(
            metrics[KITTIDepthEvaluator.KEY_RMSE], 0.0, places=3
        )
        self.assertAlmostEqual(
            metrics[KITTIDepthEvaluator.KEY_RMSE_LOG], 0.0, places=3
        )
        self.assertAlmostEqual(
            metrics[KITTIDepthEvaluator.KEY_ABS_ERR], 0.0, places=3
        )
        self.assertAlmostEqual(
            metrics[KITTIDepthEvaluator.KEY_ABS_REL], 0.0, places=3
        )
        self.assertAlmostEqual(
            metrics[KITTIDepthEvaluator.KEY_SQ_REL], 0.0, places=3
        )
        self.assertAlmostEqual(
            metrics[KITTIDepthEvaluator.KEY_DELTA05], 1.0, places=3
        )
        self.assertAlmostEqual(
            metrics[KITTIDepthEvaluator.KEY_DELTA1], 1.0, places=3
        )
        self.assertAlmostEqual(
            metrics[KITTIDepthEvaluator.KEY_DELTA2], 1.0, places=3
        )
        self.assertAlmostEqual(
            metrics[KITTIDepthEvaluator.KEY_DELTA3], 1.0, places=3
        )

    def test_precomputed(self) -> None:
        """Numerical tests with precomputed values."""
        # All ones
        pred, gt = get_test_metrics()
        self.evaluator.reset()
        self.evaluator.process_batch(pred, gt)
        metrics, log_str = self.evaluator.evaluate(
            KITTIDepthEvaluator.METRIC_DEPTH
        )
        self.assertAlmostEqual(
            metrics[KITTIDepthEvaluator.KEY_SILOG], 107.738, places=3
        )
        self.assertAlmostEqual(
            metrics[KITTIDepthEvaluator.KEY_RMSE], 3.860, places=3
        )
        self.assertAlmostEqual(
            metrics[KITTIDepthEvaluator.KEY_RMSE_LOG], 1.144, places=3
        )
        self.assertAlmostEqual(
            metrics[KITTIDepthEvaluator.KEY_ABS_ERR], 3.375, places=3
        )
        self.assertAlmostEqual(
            metrics[KITTIDepthEvaluator.KEY_ABS_REL], 1.208, places=3
        )
        self.assertAlmostEqual(
            metrics[KITTIDepthEvaluator.KEY_SQ_REL], 5.635, places=3
        )
        assert isinstance(log_str, str)
