"""Test cases for visualizer callback."""
import shutil
import tempfile
import unittest

import torch

from tests.util import MockModel
from vis4d.config.default.data_connectors import CONN_BBOX_2D_VIS
from vis4d.data.const import CommonKeys as K
from vis4d.engine.callbacks import TrainerState, VisualizerCallback
from vis4d.engine.connectors import DataConnector
from vis4d.vis.image import BoundingBoxVisualizer


class TestVisualizerCallback(unittest.TestCase):
    """Test cases for callback functions."""

    def setUp(self) -> None:
        """Creates a tmp directory and setup callback."""
        self.test_dir = tempfile.mkdtemp()

        self.callback = VisualizerCallback(
            visualizer=BoundingBoxVisualizer(),
            save_prefix=self.test_dir,
            train_connector=CONN_BBOX_2D_VIS,
            test_connector=CONN_BBOX_2D_VIS,
        )

        self.callback.setup()

        self.trainer_state = TrainerState(
            current_epoch=0,
            num_epochs=0,
            global_step=0,
            data_connector=DataConnector(),
        )

    def tearDown(self) -> None:
        """Removes the tmp directory after the test."""
        shutil.rmtree(self.test_dir)

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
                K.original_images: [torch.zeros((32, 32, 3))],
                K.sample_names: ["0000"],
            },
            batch_idx=0,
        )

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
                K.original_images: [torch.zeros((32, 32, 3))],
                K.sample_names: ["0000"],
            },
            batch_idx=0,
        )