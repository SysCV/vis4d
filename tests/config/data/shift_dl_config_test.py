"""Test cases for dataset and dataloader using COCO."""
from __future__ import annotations

import unittest

from ml_collections import ConfigDict
from torch.utils.data.dataloader import DataLoader

from tests.util import get_test_data
from vis4d.config.base.datasets.shift.tasks import (
    get_shift_segmentation_config,
)
from vis4d.config.util import class_config, instantiate_classes
from vis4d.data.io import HDF5Backend


class TestMultiViewDataloaderConfig(unittest.TestCase):
    """Test cases for the dataloader config for multiview dataset."""

    DATA_ROOT = get_test_data("shift_test")
    VIEWS = ["front"]
    SPLIT = "val"
    DOMAIN_ATTR = [{"weather_coarse": "rainy", "timeofday_coarse": "night"}]

    def test_dataloader_config(self) -> None:
        """Test case to instantiate a dataloader from a config.

        This also checks that the detection preprocessing works.
        """
        dataloader_cfg = get_shift_segmentation_config(
            data_root=self.DATA_ROOT,
            train_split=self.SPLIT,
            test_split=self.SPLIT,
            train_views_to_load=self.VIEWS,
            test_views_to_load=self.VIEWS,
            train_attributes_to_load=self.DOMAIN_ATTR,
            test_attributes_to_load=self.DOMAIN_ATTR,
            data_backend=class_config(HDF5Backend),
            workers_per_gpu=0,
        )
        self.assertTrue(isinstance(dataloader_cfg, ConfigDict))
        train_dl = instantiate_classes(dataloader_cfg.train_dataloader)
        self.assertTrue(isinstance(train_dl, DataLoader))
        entries = next(iter(train_dl))
        self.assertEqual(entries["images"].shape, (1, 3, 512, 512))
        self.assertEqual(entries["seg_masks"].shape, (1, 512, 512))
