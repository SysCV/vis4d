"""SHIFT eval test cases."""

from __future__ import annotations

import unittest

import numpy as np

from tests.eval.utils import get_dataloader
from tests.util import get_test_data
from vis4d.data.datasets.shift import SHIFT
from vis4d.eval.shift import SHIFTOpticalFlowEvaluator


class TestSegEvaluator(unittest.TestCase):
    """Tests for SegEvaluator."""

    evaluator = SHIFTOpticalFlowEvaluator()
    dataset = SHIFT(
        data_root=get_test_data("shift_test"),
        split="val",
        keys_to_load=["images", "boxes2d", "optical_flows"],
    )
    test_loader = get_dataloader(dataset, 1, sensors=["front"])

    def test_shift_prediction(self) -> None:
        """Tests using shift data."""
        for batch in self.test_loader:
            gts = batch["front"]["optical_flows"]
            preds = np.zeros((1, 800, 1280, 2))
            self.evaluator.process_batch(prediction=preds, groundtruth=gts)

        metrics, _ = self.evaluator.evaluate(
            SHIFTOpticalFlowEvaluator.METRIC_FLOW
        )
        self.assertAlmostEqual(metrics["EPE"], 8.6018, places=3)

    def test_shift_perfect_prediction(self) -> None:
        """Tests when predictions are correct."""
        for batch in self.test_loader:
            gts = batch["front"]["optical_flows"]
            preds = gts
            self.evaluator.process_batch(prediction=preds, groundtruth=gts)

        metrics, _ = self.evaluator.evaluate(
            SHIFTOpticalFlowEvaluator.METRIC_FLOW
        )
        self.assertAlmostEqual(metrics["EPE"], 0.0, places=4)
