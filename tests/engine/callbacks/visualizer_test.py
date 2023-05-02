"""Test cases for visualizer callback."""
import tempfile
import unittest

import torch

from tests.util import MockModel
from vis4d.config.default.data_connectors import CONN_BBOX_2D_VIS
from vis4d.engine.callbacks import TrainerState, VisualizerCallback
from vis4d.engine.connectors import DataConnector
from vis4d.vis.image import BoundingBoxVisualizer


class TestVisualizerCallback(unittest.TestCase):
    """Test cases for callback functions."""

    callback = VisualizerCallback(
        visualizer=BoundingBoxVisualizer(),
        save_prefix=tempfile.mkdtemp(),
        save_to_disk=True,
        train_connector=CONN_BBOX_2D_VIS,
        test_connector=CONN_BBOX_2D_VIS,
    )

    trainer_state = TrainerState(
        current_epoch=0,
        num_epochs=0,
        global_step=0,
        data_connector=DataConnector(),
    )

    def test_setup(self):
        """Test set_up function."""
        self.callback.setup()

    def test_on_train_epoch_start(self) -> None:
        """Test on_train_epoch_start function."""
        self.callback.on_train_epoch_start(self.trainer_state, MockModel(0))

    def test_on_train_batch_end(self) -> None:
        """Test on_train_batch_end function."""
        self.callback.on_train_batch_end(
            self.trainer_state,
            MockModel(0),
            outputs={
                "boxes": [torch.zeros((0, 4))],
                "scores": [torch.zeros((0,))],
                "class_ids": [torch.zeros((0,))],
            },
            batch={
                "original_images": [torch.zeros((32, 32, 3))],
                "sample_names": ["0000"],
            },
            batch_idx=0,
        )

    def test_on_test_epoch_start(self) -> None:
        """Test on_test_epoch_start function."""
        self.callback.on_test_epoch_start(self.trainer_state, MockModel(0))

    def test_on_test_batch_end(self) -> None:
        """Test the visualizer callback."""
        self.callback.on_test_batch_end(
            self.trainer_state,
            MockModel(0),
            outputs={
                "boxes": [torch.zeros((0, 4))],
                "scores": [torch.zeros((0,))],
                "class_ids": [torch.zeros((0,))],
            },
            batch={
                "original_images": [torch.zeros((32, 32, 3))],
                "sample_names": ["0000"],
            },
            batch_idx=0,
        )
