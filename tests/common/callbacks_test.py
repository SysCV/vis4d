"""Test cases for progress bar."""
import tempfile
import unittest

import torch

from vis4d.common.callbacks import (
    CheckpointCallback,
    EvaluatorCallback,
    LoggingCallback,
    VisualizerCallback,
)
from vis4d.eval.detect.coco import COCOEvaluator
from vis4d.vis.image import BoundingBoxVisualizer

from ..util import MockModel, get_test_data


class TestCallbacks(unittest.TestCase):
    """Test cases for callback functions."""

    def test_evaluator_callback(self) -> None:
        """Test the evaluator callback."""
        clbk = EvaluatorCallback(
            evaluator=COCOEvaluator(
                data_root=get_test_data("coco_test"), split="train"
            ),
            save_prefix=tempfile.mkdtemp(),
            run_every_nth_epoch=1,
            num_epochs=5,
        )
        clbk.on_train_batch_end(MockModel(0), {}, {})
        clbk.on_train_epoch_end(MockModel(0), 0)
        clbk.on_test_batch_end(
            MockModel(0),
            {},
            {
                "coco_image_id": [0],
                "pred_boxes": [torch.zeros((0, 4))],
                "pred_scores": [torch.zeros((0, 1))],
                "pred_classes": [torch.zeros((0, 1))],
            },
        )
        clbk.on_test_epoch_end(MockModel(0), 0)

    def test_visualizer_callback(self) -> None:
        """Test the visualizer callback."""
        clbk = VisualizerCallback(
            visualizer=BoundingBoxVisualizer(),
            save_prefix=tempfile.mkdtemp(),
            run_every_nth_epoch=1,
            num_epochs=5,
        )
        clbk.on_train_batch_end(MockModel(0), {}, {})
        clbk.on_train_epoch_end(MockModel(0), 0)
        clbk.on_test_batch_end(
            MockModel(0),
            {},
            {
                "images": [torch.zeros((32, 32, 3))],
                "boxes": [torch.zeros((0, 4))],
            },
        )
        clbk.on_test_epoch_end(MockModel(0), 0)

    def test_logging_callback(self) -> None:
        """Test the logging callback."""
        clbk = LoggingCallback(refresh_rate=1)
        clbk.on_train_batch_end(
            MockModel(0),
            {
                "metrics": {"loss1": 0, "loss2": 1},
                "epoch": 0,
                "cur_iter": 0,
                "total_iters": 1,
            },
            {},
        )
        clbk.on_train_epoch_end(MockModel(0), 0)
        clbk.on_test_batch_end(
            MockModel(0),
            {
                "cur_iter": 0,
                "total_iters": 1,
            },
            {},
        )
        clbk.on_test_epoch_end(MockModel(0), 0)

    def test_checkpoint_callback(self) -> None:
        """Test the checkpoint callback."""
        clbk = CheckpointCallback(
            save_prefix=tempfile.mkdtemp(),
            run_every_nth_epoch=1,
            num_epochs=5,
        )
        clbk.setup()
        clbk.on_train_batch_end(MockModel(0), {}, {})
        clbk.on_train_epoch_end(MockModel(0), 0)
        clbk.on_test_batch_end(MockModel(0), {}, {})
        clbk.on_test_epoch_end(MockModel(0), 0)
