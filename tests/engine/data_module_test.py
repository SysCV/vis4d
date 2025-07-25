"""Pytorch lightning utilities for unit tests."""

from __future__ import annotations

import unittest

from torch.utils.data.dataloader import DataLoader

from tests.util import get_test_data
from vis4d.engine.data_module import DataModule
from vis4d.zoo.base.datasets.coco import get_coco_detection_cfg


class DataModuleTest(unittest.TestCase):
    """Pytorch lightning data module test class."""

    def setUp(self) -> None:
        """Set up the test case."""
        dataloader_cfg = get_coco_detection_cfg(
            get_test_data("coco_test"),
            train_split="train",
            test_split="train",
            cache_as_binary=False,
        )
        self.datamodule = DataModule(dataloader_cfg)

    def test_train_dataloader(self) -> None:
        """Test train dataloader."""
        self.assertTrue(
            isinstance(self.datamodule.train_dataloader(), DataLoader)
        )

    def test_test_dataloader(self) -> None:
        """Test test dataloader."""
        self.assertTrue(isinstance(self.datamodule.test_dataloader(), list))
        for dl in self.datamodule.test_dataloader():
            self.assertTrue(isinstance(dl, DataLoader))

    def test_val_dataloader(self) -> None:
        """Test val dataloader."""
        self.assertTrue(isinstance(self.datamodule.val_dataloader(), list))
        for dl in self.datamodule.val_dataloader():
            self.assertTrue(isinstance(dl, DataLoader))
