"""Testcases for COCO evaluator."""

from __future__ import annotations

import unittest

import torch

from tests.engine.trainer_test import get_test_dataloader
from tests.util import generate_boxes, get_test_data
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets import COCO
from vis4d.eval.coco import COCODetectEvaluator


class TestCOCODetectEvaluator(unittest.TestCase):
    """COCO evaluator testcase class."""

    def test_coco_eval(self) -> None:
        """Testcase for COCO evaluation."""
        batch_size = 2
        coco_metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
        coco_eval = COCODetectEvaluator(
            get_test_data("coco_test"), split="train", per_class_eval=True
        )

        # test empty
        boxes, scores, classes, _ = generate_boxes(512, 512, 20, batch_size)
        coco_eval.process_batch([37777, 397133], boxes, scores, classes)
        coco_eval.process()
        assert coco_eval.metrics == ["Det", "InsSeg"]
        score_dict, log_str = coco_eval.evaluate(coco_eval.METRIC_DET)
        for metric in coco_metrics:
            assert metric in score_dict
            assert score_dict[metric] < 0.02
        assert isinstance(log_str, str)
        coco_eval.reset()

        # test gt
        dataset = COCO(
            get_test_data("coco_test"),
            keys_to_load=(K.images, K.boxes2d, K.boxes2d_classes),
            split="train",
        )
        test_loader = get_test_dataloader(dataset, batch_size)[0]
        batch = next(iter(test_loader))
        coco_eval.process_batch(
            batch[K.sample_names],
            batch[K.boxes2d],
            [torch.ones((len(b), 1)) for b in batch[K.boxes2d]],
            batch[K.boxes2d_classes],
        )
        coco_eval.process()
        score_dict, log_str = coco_eval.evaluate(coco_eval.METRIC_DET)
        for metric in coco_metrics:
            assert metric in score_dict
            assert score_dict[metric] > 0.99
        assert isinstance(log_str, str)
