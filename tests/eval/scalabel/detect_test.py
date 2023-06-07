"""Testcases for BDD100K tracking evaluator."""
from __future__ import annotations

import os.path as osp
import unittest

from tests.util import generate_boxes, generate_instance_masks, get_test_data
from vis4d.data.datasets.shift import SHIFT
from vis4d.engine.connectors import (
    data_key,
    get_inputs_for_pred_and_data,
    pred_key,
)
from vis4d.eval.scalabel import ScalabelDetectEvaluator

from ..utils import get_dataloader


class TestBDD100KTrackEvaluator(unittest.TestCase):
    """BDD100K tracking evaluator testcase class."""

    CONN_SHIFT_EVAL = {
        "frame_ids": data_key("frame_ids"),
        "sample_names": data_key("sample_names"),
        "sequence_names": data_key("sequence_names"),
        "pred_boxes": pred_key("boxes"),
        "pred_classes": pred_key("class_ids"),
        "pred_scores": pred_key("scores"),
        "pred_masks": pred_key("masks"),
    }

    def test_shift_eval(self) -> None:
        """Testcase for SHIFT evaluation."""
        batch_size = 1

        annotations = osp.join(
            get_test_data("shift_test"),
            "discrete/images/val/front/det_2d.json",
        )
        scalabel_eval = ScalabelDetectEvaluator(annotation_path=annotations)
        assert str(scalabel_eval) == "Scalabel Tracking Evaluator"
        assert scalabel_eval.metrics == ["Det", "InsSeg"]

        # test gt
        dataset = SHIFT(data_root=get_test_data("shift_test"), split="val")
        test_loader = get_dataloader(dataset, batch_size)

        boxes, scores, classes, track_ids = generate_boxes(
            800, 1280, 4, batch_size, True
        )
        masks, _, _ = generate_instance_masks(800, 1280, 4, batch_size)
        output = {
            "boxes": boxes,
            "scores": scores,
            "class_ids": classes,
            "track_ids": track_ids,
            "masks": masks,
        }

        for batch in test_loader:
            scalabel_eval.process_batch(
                **get_inputs_for_pred_and_data(
                    self.CONN_SHIFT_EVAL, output, batch
                )
            )

        _, log_str = scalabel_eval.evaluate("Det")
        assert isinstance(log_str, str)
        assert log_str.count("\n") == 12

        _, log_str = scalabel_eval.evaluate("InsSeg")
        assert isinstance(log_str, str)
        assert log_str.count("\n") == 12
