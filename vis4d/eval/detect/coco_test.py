"""Testcases for COCO evaluator."""
from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from vis4d.data.const import CommonKeys
from vis4d.data.datasets import COCO
from vis4d.data.loader import DataPipe, build_inference_dataloaders
from vis4d.unittest.util import generate_boxes, get_test_data

from .coco import COCOEvaluator


def get_dataloader(datasets: Dataset, batch_size: int) -> DataLoader:
    """Get data loader for testing."""
    datapipe = DataPipe(datasets)
    return build_inference_dataloaders(
        datapipe, samples_per_gpu=batch_size, workers_per_gpu=1
    )[0]


class TestCOCOEvaluator(unittest.TestCase):
    """COCO evaluator testcase class."""

    def test_coco_eval(self) -> None:
        """Testcase for COCO evaluation."""
        batch_size = 2
        coco_metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
        coco_eval = COCOEvaluator(
            get_test_data("coco_test"), split="train", per_class_eval=True
        )

        # test empty
        boxes, scores, classes, _ = generate_boxes(512, 512, 20, batch_size)
        model_output = {
            "boxes2d": boxes,
            "boxes2d_scores": scores,
            "boxes2d_classes": classes,
        }
        coco_eval.process({"coco_image_id": [37777, 397133]}, model_output)
        score_dict, log_str = coco_eval.evaluate("COCO_AP")
        for metric in coco_metrics:
            assert metric in score_dict
            assert score_dict[metric] < 0.02
        assert isinstance(log_str, str)
        coco_eval.reset()

        # test gt
        dataset = COCO(
            get_test_data("coco_test"),
            keys=(CommonKeys.boxes2d, CommonKeys.boxes2d_classes),
            split="train",
        )
        test_loader = get_dataloader(dataset, batch_size)
        batch = next(iter(test_loader))
        model_output = {
            "boxes2d": batch["boxes2d"],
            "boxes2d_scores": [
                torch.ones((len(b), 1)) for b in batch["boxes2d"]
            ],
            "boxes2d_classes": batch["boxes2d_classes"],
        }
        coco_eval.process(batch, model_output)
        score_dict, log_str = coco_eval.evaluate("COCO_AP")
        for metric in coco_metrics:
            assert metric in score_dict
            assert score_dict[metric] > 0.98
        assert isinstance(log_str, str)
