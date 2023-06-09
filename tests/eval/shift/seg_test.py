"""SHIFT eval test cases."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from tests.eval.utils import get_dataloader
from tests.util import get_test_data
from vis4d.data.datasets.shift import SHIFT
from vis4d.eval.shift import SHIFTSegEvaluator


class TestSegEvaluator(unittest.TestCase):
    """Tests for SegEvaluator."""

    evaluator = SHIFTSegEvaluator(ignore_classes_as_cityscapes=True)
    dataset = SHIFT(
        data_root=get_test_data("shift_test"),
        split="val",
        keys_to_load=["images", "boxes2d", "seg_masks"],
    )
    test_loader = get_dataloader(dataset, 2)

    def test_shift_prediction(self) -> None:
        """Tests using shift data."""
        for batch in self.test_loader:
            gts = batch["front"][0]["seg_masks"].unsqueeze(0)
            preds = np.zeros((1, 23, 800, 1280))
            preds[:, 2, :, :] = 1
            self.evaluator.process_batch(prediction=preds, groundtruth=gts)

        metrics, _ = self.evaluator.evaluate("mIoU")
        self.assertAlmostEqual(metrics["mIoU"], 46.46, places=2)

    def test_shift_perfect_prediction(self) -> None:
        """Tests when predictions are correct."""
        for batch in self.test_loader:
            gts = batch["front"][0]["seg_masks"].unsqueeze(0)
            preds = (
                torch.eye(23)[gts.squeeze(0).long()]
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            self.evaluator.process_batch(prediction=preds, groundtruth=gts)

        metrics, _ = self.evaluator.evaluate("mIoU")
        self.assertAlmostEqual(metrics["mIoU"], 100.0, places=2)
