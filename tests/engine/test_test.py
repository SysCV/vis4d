"""Engine test tests."""
from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from tests.util import get_test_data
from vis4d.common.callbacks import LoggingCallback
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets import COCO
from vis4d.data.loader import DataPipe, build_inference_dataloaders
from vis4d.data.transforms import normalize, resize
from vis4d.data.transforms.base import compose
from vis4d.engine.connectors import DataConnectionInfo, StaticDataConnector
from vis4d.engine.test import Tester
from vis4d.model.segment.semantic_fpn import SemanticFPN


def get_test_dataloader(datasets: Dataset) -> DataLoader:
    """Get data loader for testing."""
    preprocess_fn = compose(
        [
            resize.GenerateResizeParameters((64, 64)),
            resize.ResizeImage(),
            normalize.NormalizeImage(),
        ]
    )
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_inference_dataloaders(
        datapipe, samples_per_gpu=1, workers_per_gpu=1
    )[0]


class EngineTestTest(unittest.TestCase):
    """Engine test test class."""

    def test_test(self) -> None:
        """Test engine testing."""
        state = torch.random.get_rng_state()
        torch.random.set_rng_state(torch.manual_seed(0).get_state())

        model = SemanticFPN(num_classes=80)
        dataset = COCO(get_test_data("coco_test"), split="train")
        test_loader = get_test_dataloader(dataset)
        data_connector = StaticDataConnector(
            connections=DataConnectionInfo(
                test={K.images: K.images, "original_hw": "original_hw"}
            )
        )
        callback = {"logger": LoggingCallback(1)}
        tester = Tester([test_loader], data_connector, callback)

        model.eval()
        tester.test(model, epoch=0)

        torch.random.set_rng_state(state)
