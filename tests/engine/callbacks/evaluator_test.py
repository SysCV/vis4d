"""Test cases for evaluator callback."""
import tempfile
import unittest

import torch

from tests.util import MockModel, get_test_data
from vis4d.config.base.datasets.coco_detection import CONN_COCO_BBOX_EVAL
from vis4d.engine.callbacks import EvaluatorCallback, TrainerState
from vis4d.engine.connectors import DataConnector
from vis4d.eval.detect.coco import COCOEvaluator


class TestEvaluatorCallback(unittest.TestCase):
    """Test cases for callback functions."""

    callback = EvaluatorCallback(
        evaluator=COCOEvaluator(
            data_root=get_test_data("coco_test"), split="train"
        ),
        save_predictions=True,
        save_prefix=tempfile.mkdtemp(),
        test_connector=CONN_COCO_BBOX_EVAL,
    )

    trainer_state = TrainerState(
        current_epoch=0,
        num_epochs=0,
        global_step=0,
        data_connector=DataConnector(),
    )

    def test_on_test_batch_end(self) -> None:
        """Test on_test_batch_end function."""
        self.callback.on_test_batch_end(
            self.trainer_state,
            MockModel(0),
            outputs={
                "boxes": [torch.zeros((0, 4))],
                "scores": [torch.zeros((0, 1))],
                "class_ids": [torch.zeros((0, 1))],
            },
            batch={"coco_image_id": [0]},
            batch_idx=0,
        )

    def test_on_test_epoch_end(self) -> None:
        """Test on_test_epoch_end function."""
        self.callback.on_test_epoch_end(self.trainer_state, MockModel(0))
