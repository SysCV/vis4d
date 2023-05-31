"""SHIFT eval test cases."""
from __future__ import annotations

import unittest

import numpy as np

from tests.eval.utils import get_dataloader
from tests.util import get_test_data
from vis4d.data.datasets.shift import SHIFT
from vis4d.eval.shift import SHIFTDepthEvaluator


class TestSegEvaluator(unittest.TestCase):
    """Tests for SegEvaluator."""

    evaluator = SHIFTDepthEvaluator(use_eval_crop=True)
    dataset = SHIFT(
        data_root=get_test_data("shift_test"),
        split="val",
        keys_to_load=["images", "boxes2d", "depth_maps"],
    )
    test_loader = get_dataloader(dataset, 1, sensors=["front"])

    def test_shift_prediction(self) -> None:
        """Tests using shift data."""
        for batch in self.test_loader:
            gts = batch["front"]["depth_maps"]
            preds = np.zeros((1, 800, 1280))
            self.evaluator.process_batch(prediction=preds, groundtruth=gts)

        metrics, _ = self.evaluator.evaluate(SHIFTDepthEvaluator.METRIC_DEPTH)
        self.assertAlmostEqual(metrics["AbsErr"], 7.734, places=2)

    def test_shift_perfect_prediction(self) -> None:
        """Tests when predictions are correct."""
        for batch in self.test_loader:
            gts = batch["front"]["depth_maps"]
            preds = gts
            self.evaluator.process_batch(prediction=preds, groundtruth=gts)

        metrics, _ = self.evaluator.evaluate(SHIFTDepthEvaluator.METRIC_DEPTH)
        self.assertAlmostEqual(metrics["AbsErr"], 0.0, places=2)
