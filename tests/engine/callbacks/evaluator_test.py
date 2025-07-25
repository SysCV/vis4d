"""Test cases for evaluator callback."""

import shutil
import tempfile
import unittest

import lightning.pytorch as pl
import torch

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as K
from vis4d.engine.callbacks import EvaluatorCallback
from vis4d.engine.connectors import CallbackConnector
from vis4d.eval.coco import COCODetectEvaluator
from vis4d.zoo.base.datasets.coco import CONN_COCO_MASK_EVAL


class TestEvaluatorCallback(unittest.TestCase):
    """Test cases for callback functions."""

    def setUp(self) -> None:
        """Creates a tmp directory and setup callback."""
        self.test_dir = tempfile.mkdtemp()

        self.trainer = pl.Trainer()
        self.training_module = pl.LightningModule()

        self.callback = EvaluatorCallback(
            evaluator=COCODetectEvaluator(
                data_root=get_test_data("coco_test"), split="train"
            ),
            save_predictions=True,
            metrics_to_eval=[COCODetectEvaluator.METRIC_DET],
            output_dir=self.test_dir,
            test_connector=CallbackConnector(CONN_COCO_MASK_EVAL),
        )

        self.callback.setup(self.trainer, self.training_module, stage="test")

    def tearDown(self) -> None:
        """Removes the tmp directory after the test."""
        shutil.rmtree(self.test_dir)

    def test_evaluator_callback(self) -> None:
        """Test evaluator callback function."""
        self.callback.on_test_batch_end(
            self.trainer,
            self.training_module,
            outputs={
                "boxes": {
                    "boxes": [torch.zeros((2, 4))],
                    "scores": [torch.zeros((2, 1))],
                    "class_ids": [torch.zeros((2, 1))],
                },
                "masks": [torch.zeros((2, 10, 10))],
            },
            batch={K.sample_names: [397133]},
            batch_idx=0,
        )

        self.callback.on_test_epoch_end(self.trainer, self.training_module)
