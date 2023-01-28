"""Test cases for dataset and dataloader using COCO."""
from __future__ import annotations

import unittest

from ml_collections import ConfigDict
from torch.utils.data.dataloader import DataLoader

from tests.util import get_test_data
from vis4d.config.default.data.dataloader import default_dataloader_config
from vis4d.config.default.data.detect import default_detection_preprocessing
from vis4d.config.util import class_config, instantiate_classes
from vis4d.data.datasets.coco import COCO


class TestDataloaderConfig(unittest.TestCase):
    """Test cases for the dataloader config."""

    COCO_DATA_ROOT = get_test_data("coco_test")

    def test_dataset_config_str(self) -> None:
        """Test case to instantiate a dataset from a string."""
        # Check full path imports
        dataset_cfg_train = class_config(
            "vis4d.data.datasets.coco.COCO",
            data_root=self.COCO_DATA_ROOT,
            split="train",
        )
        self.assertTrue(isinstance(dataset_cfg_train, ConfigDict))
        coco = instantiate_classes(dataset_cfg_train)
        self.assertTrue(isinstance(coco, COCO))
        self.assertEqual(coco.data_root, self.COCO_DATA_ROOT)
        # Make sure it is callable. I.e. does not crash
        _ = next(iter(coco))

    def test_dataset_config_clazz(self) -> None:
        """Test case to instantiate a dataset from a class."""
        # Check full path imports
        dataset_cfg_train = class_config(
            COCO,
            data_root=self.COCO_DATA_ROOT,
            split="train",
        )
        self.assertTrue(isinstance(dataset_cfg_train, ConfigDict))
        coco = instantiate_classes(dataset_cfg_train)
        self.assertTrue(isinstance(coco, COCO))
        self.assertEqual(coco.data_root, self.COCO_DATA_ROOT)
        # Make sure it is callable. I.e. does not crash
        _ = next(iter(coco))

    def test_dataloader_config(self) -> None:
        """Test case to instantiate a dataloader from a config.

        This also checks that the detection preprocessing works.
        """
        dataset_cfg_train = class_config(
            COCO,
            data_root=self.COCO_DATA_ROOT,
            split="train",
        )

        preprocess_cfg_train = default_detection_preprocessing(
            800, 1333, augmentation_probability=0.5
        )
        dataloader_train_cfg = default_dataloader_config(
            preprocess_cfg_train,
            dataset_cfg_train,
            2,
            2,
            batchprocess_fn=class_config(
                "vis4d.data.transforms.pad.pad_image"
            ),
        )
        self.assertTrue(isinstance(dataloader_train_cfg, ConfigDict))
        dl = instantiate_classes(dataloader_train_cfg)
        self.assertTrue(isinstance(dl, DataLoader))
        entries = next(iter(dl))
        self.assertEqual(entries["images"].shape[0], 2)
