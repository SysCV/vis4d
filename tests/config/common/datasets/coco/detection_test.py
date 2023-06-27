"""Test cases for dataset and dataloader using COCO."""
from __future__ import annotations

import unittest

from ml_collections import ConfigDict
from torch.utils.data.dataloader import DataLoader

from tests.util import get_test_data
from vis4d.config import class_config, instantiate_classes
from vis4d.config.common.datasets.coco import get_coco_detection_cfg
from vis4d.data.datasets.coco import COCO


class TestDataloaderConfig(unittest.TestCase):
    """Test cases for the dataloader config."""

    COCO_DATA_ROOT = get_test_data("coco_test")

    def test_dataset_config_str(self) -> None:
        """Test case to instantiate a dataset from a string."""
        # Check full path imports
        train_dataset_cfg = class_config(
            "vis4d.data.datasets.coco.COCO",
            data_root=self.COCO_DATA_ROOT,
            split="train",
        )
        self.assertTrue(isinstance(train_dataset_cfg, ConfigDict))
        coco = instantiate_classes(train_dataset_cfg)
        self.assertTrue(isinstance(coco, COCO))
        self.assertEqual(coco.data_root, self.COCO_DATA_ROOT)
        # Make sure it is callable. I.e. does not crash
        _ = next(iter(coco))

    def test_dataset_config_clazz(self) -> None:
        """Test case to instantiate a dataset from a class."""
        train_dataset_cfg = class_config(
            COCO,
            data_root=self.COCO_DATA_ROOT,
            split="train",
        )
        self.assertTrue(isinstance(train_dataset_cfg, ConfigDict))
        coco = instantiate_classes(train_dataset_cfg)
        self.assertTrue(isinstance(coco, COCO))
        self.assertEqual(coco.data_root, self.COCO_DATA_ROOT)
        # Make sure it is callable. I.e. does not crash
        _ = next(iter(coco))

    def test_dataloader_config(self) -> None:
        """Test case to instantiate a dataloader from a config.

        This also checks that the detection preprocessing works.
        """
        dataloader_cfg = get_coco_detection_cfg(
            self.COCO_DATA_ROOT,
            train_split="train",
            test_split="train",
            cache_as_binary=False,
        )
        self.assertTrue(isinstance(dataloader_cfg, ConfigDict))
        train_dl = instantiate_classes(dataloader_cfg.train_dataloader)
        self.assertTrue(isinstance(train_dl, DataLoader))
        entries = next(iter(train_dl))
        self.assertEqual(entries["images"].shape[0], 2)
