"""Test cases for progress bar."""
import tempfile
import unittest

import torch

from vis4d.common.callbacks import (
    CheckpointCallback,
    EvaluatorCallback,
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
            output_dir=tempfile.mkdtemp(),
            run_every_nth_epoch=1,
            num_epochs=5,
        )
        clbk.on_train_batch_end()
        clbk.on_train_epoch_end(MockModel(0), 0)
        clbk.on_test_batch_end(
            {
                "coco_image_id": [0],
                "pred_boxes": [torch.zeros((0, 4))],
                "pred_scores": [torch.zeros((0, 1))],
                "pred_classes": [torch.zeros((0, 1))],
            }
        )
        clbk.on_test_epoch_end()

    def test_visualizer_callback(self) -> None:
        """Test the visualizer callback."""
        clbk = VisualizerCallback(
            visualizer=BoundingBoxVisualizer(),
            output_dir=tempfile.mkdtemp(),
            run_every_nth_epoch=1,
            num_epochs=5,
        )
        clbk.on_train_batch_end()
        clbk.on_train_epoch_end(MockModel(0), 0)
        clbk.on_test_batch_end(
            {
                "images": [torch.zeros((32, 32, 3))],
                "boxes": [torch.zeros((0, 4))],
            }
        )
        clbk.on_test_epoch_end()

    def test_checkpoint_callback(self) -> None:
        """Test the checkpoint callback."""
        clbk = CheckpointCallback(
            save_prefix=tempfile.mkdtemp(),
            run_every_nth_epoch=1,
            num_epochs=5,
        )
        clbk.on_train_batch_end()
        clbk.on_train_epoch_end(MockModel(0), 0)
        clbk.on_test_batch_end({})
        clbk.on_test_epoch_end()
