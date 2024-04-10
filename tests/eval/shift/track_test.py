"""SHIFT eval test cases."""

from __future__ import annotations

import unittest

import torch

from tests.eval.utils import get_dataloader
from tests.util import get_test_data
from vis4d.data.datasets.shift import SHIFT
from vis4d.eval.shift import SHIFTTrackEvaluator


class TestSegEvaluator(unittest.TestCase):
    """Tests for SegEvaluator."""

    data_root = get_test_data("shift_test")
    evaluator = SHIFTTrackEvaluator(
        annotation_path=(
            f"{data_root}/discrete/images/val/front/det_insseg_2d.json"
        )
    )
    dataset = SHIFT(
        data_root=data_root,
        split="val",
        keys_to_load=[
            "images",
            "boxes2d",
            "boxes2d_classes",
            "boxes2d_track_ids",
            "instance_masks",
        ],
    )
    test_loader = get_dataloader(dataset, 1, sensors=["front"])

    def test_shift_perfect_prediction(self) -> None:
        """Tests when predictions are correct."""
        for batch in self.test_loader:
            self.evaluator.process_batch(
                frame_ids=batch["frame_ids"],
                sample_names=batch["sample_names"],
                sequence_names=batch["sequence_names"],
                pred_boxes=batch["front"]["boxes2d"],
                pred_classes=batch["front"]["boxes2d_classes"],
                pred_scores=[
                    torch.ones_like(batch["front"]["boxes2d_classes"][0])
                ],
                pred_track_ids=batch["front"]["boxes2d_track_ids"],
                pred_masks=batch["front"]["instance_masks"],
            )

        metrics, _ = self.evaluator.evaluate("MOT")
        self.assertAlmostEqual(metrics["MOTA"], 100.0, places=2)
        self.assertAlmostEqual(metrics["MOTP"], 100.0, places=2)

        metrics, _ = self.evaluator.evaluate("MOTS")
        self.assertAlmostEqual(metrics["MOTA"], 100.0, places=2)
        self.assertAlmostEqual(metrics["MOTP"], 100.0, places=2)
