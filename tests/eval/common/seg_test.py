"""Seg eval test cases."""
from __future__ import annotations

import unittest

import numpy as np

from vis4d.common.typing import NDArrayNumber
from vis4d.eval.common import SegEvaluator


def get_test_data() -> tuple[NDArrayNumber, NDArrayNumber]:
    """Precomputed input data."""
    pred = np.asarray(
        [
            [8, 3, 5, 4],
            [8, 3, 2, 0],
            [3, 1, 2, 4],
            [1, 8, 9, 0],
            [6, 8, 0, 8],
            [5, 4, 2, 6],
            [4, 0, 7, 5],
            [6, 5, 0, 1],
            [5, 0, 7, 9],
            [7, 5, 7, 7],
            [1, 0, 6, 8],
            [6, 0, 7, 7],
            [4, 8, 2, 0],
            [9, 9, 4, 9],
            [0, 2, 8, 3],
            [0, 1, 6, 9],
            [0, 7, 1, 0],
            [9, 9, 3, 3],
            [8, 2, 9, 3],
            [8, 9, 9, 1],
            [8, 7, 5, 8],
            [4, 2, 0, 5],
            [1, 6, 3, 4],
            [6, 9, 0, 2],
            [0, 8, 5, 4],
            [1, 5, 3, 6],
            [1, 3, 8, 9],
            [0, 5, 5, 1],
            [4, 5, 7, 7],
            [9, 1, 3, 7],
            [7, 1, 4, 1],
            [9, 9, 2, 7],
            [4, 1, 6, 4],
            [8, 1, 9, 1],
            [1, 1, 1, 1],
            [5, 3, 8, 7],
            [9, 7, 3, 3],
            [2, 7, 7, 8],
            [8, 1, 9, 4],
            [9, 3, 6, 3],
            [5, 0, 2, 1],
            [9, 2, 0, 8],
            [4, 9, 6, 1],
            [8, 9, 3, 9],
            [2, 5, 0, 3],
            [8, 8, 3, 8],
            [7, 2, 4, 3],
            [1, 6, 7, 0],
            [7, 1, 5, 9],
            [0, 2, 1, 4],
            [0, 3, 2, 5],
            [4, 1, 3, 9],
            [2, 7, 2, 4],
            [4, 1, 5, 2],
            [6, 5, 5, 5],
            [0, 2, 5, 2],
            [8, 0, 5, 6],
            [5, 7, 7, 8],
            [1, 1, 1, 5],
            [7, 2, 4, 4],
            [0, 7, 9, 5],
            [5, 4, 4, 5],
            [0, 4, 7, 3],
            [3, 3, 7, 4],
            [4, 1, 9, 7],
            [0, 2, 5, 7],
            [0, 9, 9, 6],
            [3, 5, 2, 5],
            [7, 1, 0, 3],
            [9, 7, 5, 3],
            [3, 7, 9, 6],
            [4, 2, 8, 5],
            [2, 8, 8, 5],
            [6, 0, 4, 0],
            [1, 1, 1, 4],
            [5, 2, 4, 8],
            [4, 3, 2, 8],
            [7, 7, 2, 5],
            [2, 4, 4, 7],
            [2, 9, 1, 1],
            [4, 2, 8, 1],
            [4, 3, 4, 1],
            [3, 5, 5, 5],
            [5, 9, 1, 2],
            [2, 6, 6, 2],
            [7, 6, 8, 9],
            [4, 6, 5, 6],
            [5, 7, 1, 0],
            [6, 9, 7, 4],
            [7, 7, 7, 2],
            [7, 5, 6, 9],
            [3, 8, 1, 4],
            [8, 9, 2, 8],
            [3, 6, 1, 3],
            [1, 7, 8, 7],
            [1, 3, 9, 1],
            [2, 6, 1, 5],
            [6, 8, 3, 5],
            [5, 9, 5, 2],
            [9, 7, 7, 7],
        ]
    )

    gt = np.asarray(
        [
            [0, 4, 4, 9],
            [5, 1, 8, 4],
            [0, 5, 8, 5],
            [3, 3, 7, 5],
            [2, 4, 0, 8],
            [7, 7, 6, 9],
            [5, 1, 4, 4],
            [5, 5, 2, 2],
            [1, 8, 5, 7],
            [4, 4, 0, 6],
            [6, 6, 1, 3],
            [6, 8, 0, 6],
            [8, 3, 5, 0],
            [9, 2, 8, 1],
            [1, 8, 8, 5],
            [1, 8, 3, 1],
            [0, 7, 0, 5],
            [5, 5, 3, 6],
            [3, 4, 5, 7],
            [5, 2, 7, 6],
            [6, 2, 5, 0],
            [6, 6, 6, 9],
            [9, 4, 0, 2],
            [9, 9, 0, 5],
            [7, 1, 5, 8],
            [5, 3, 7, 8],
            [1, 5, 7, 2],
            [5, 7, 5, 6],
            [4, 2, 8, 3],
            [5, 1, 8, 2],
            [1, 5, 2, 1],
            [7, 8, 8, 6],
            [5, 0, 7, 5],
            [0, 3, 1, 4],
            [7, 6, 3, 5],
            [5, 5, 3, 1],
            [5, 8, 4, 2],
            [4, 0, 0, 6],
            [9, 2, 7, 7],
            [4, 8, 6, 1],
            [7, 1, 7, 3],
            [5, 7, 3, 1],
            [5, 7, 5, 0],
            [7, 3, 2, 4],
            [6, 6, 0, 1],
            [6, 0, 3, 6],
            [2, 4, 4, 5],
            [9, 7, 1, 3],
            [1, 3, 8, 2],
            [5, 6, 4, 1],
            [6, 6, 0, 1],
            [6, 0, 5, 1],
            [9, 7, 0, 2],
            [3, 5, 0, 1],
            [3, 1, 9, 8],
            [2, 9, 2, 0],
            [5, 7, 9, 3],
            [2, 8, 9, 1],
            [4, 9, 7, 3],
            [6, 1, 9, 6],
            [0, 3, 0, 3],
            [7, 6, 0, 7],
            [1, 0, 2, 4],
            [8, 6, 0, 0],
            [3, 7, 6, 3],
            [2, 4, 6, 8],
            [2, 5, 7, 2],
            [8, 8, 3, 7],
            [2, 3, 6, 3],
            [0, 1, 8, 2],
            [3, 3, 7, 4],
            [8, 9, 8, 3],
            [0, 2, 5, 0],
            [9, 7, 4, 4],
            [2, 1, 9, 8],
            [1, 9, 0, 5],
            [7, 9, 2, 0],
            [5, 8, 9, 7],
            [8, 3, 1, 2],
            [4, 4, 9, 0],
            [8, 4, 5, 8],
            [2, 1, 4, 0],
            [5, 3, 8, 7],
            [3, 7, 4, 8],
            [2, 3, 2, 4],
            [2, 5, 4, 2],
            [0, 6, 5, 7],
            [4, 8, 1, 4],
            [2, 1, 8, 1],
            [8, 1, 6, 5],
            [5, 2, 8, 2],
            [0, 6, 9, 5],
            [7, 6, 5, 3],
            [6, 7, 5, 5],
            [1, 3, 9, 2],
            [3, 6, 1, 9],
            [6, 1, 1, 5],
            [0, 7, 8, 1],
            [8, 2, 3, 7],
            [4, 1, 1, 8],
        ]
    )
    return pred, gt


class TestSegEvaluator(unittest.TestCase):
    """Tests for SegEvaluator."""

    batch_size = 4
    n_points = 100
    n_classes = 10
    evaluator = SegEvaluator(num_classes=n_classes)

    def test_perfect_prediction(self) -> None:
        """Tests when predictions are correct."""
        # All ones
        pred = np.random.rand(self.batch_size, self.n_classes, self.n_points)
        pred = np.argmax(pred, axis=1)
        gt = pred.copy()
        self.evaluator.reset()
        self.evaluator.process_batch(pred, gt)
        data, _ = self.evaluator.evaluate(SegEvaluator.METRIC_MIOU)
        self.assertEqual(data[SegEvaluator.METRIC_MIOU], 100)
        _, disc = self.evaluator.evaluate(SegEvaluator.METRIC_CONFUSION_MATRIX)
        self.assertTrue(isinstance(disc, str))

    def test_perfect_prediction_without_amax(self) -> None:
        """Tests when predictions are correct with shape [N,C,*]."""
        # All ones
        pred = np.random.rand(self.batch_size, self.n_classes, self.n_points)
        pred_amax = np.argmax(pred, axis=1)
        gt = pred_amax.copy()
        self.evaluator.reset()
        self.evaluator.process_batch(pred, gt)
        data, _ = self.evaluator.evaluate(SegEvaluator.METRIC_MIOU)
        self.assertEqual(data[SegEvaluator.METRIC_MIOU], 100)

    def test_ignore_label_prediction(self) -> None:
        """Tests when gt is all ones."""
        # All ones
        evaluator = SegEvaluator(
            num_classes=self.n_classes, class_to_ignore=255
        )
        pred = np.random.rand(self.batch_size, self.n_classes, self.n_points)
        pred = np.argmax(pred, axis=1)
        gt = pred.copy()
        pred[..., :10] = 0  # make wrong predictions
        gt[..., :10] = 255  # mask out in target

        evaluator.reset()
        evaluator.process_batch(pred, gt)
        data, _ = evaluator.evaluate(SegEvaluator.METRIC_MIOU)
        self.assertEqual(data[SegEvaluator.METRIC_MIOU], 100)

    def test_precomputed(self) -> None:
        """Numerical tests with precomputed values."""
        # All ones
        pred, gt = get_test_data()
        self.evaluator.reset()
        self.evaluator.process_batch(pred, gt)
        data, _ = self.evaluator.evaluate(SegEvaluator.METRIC_MIOU)
        self.assertEqual(data[SegEvaluator.METRIC_MIOU], 5.230499415482852)
