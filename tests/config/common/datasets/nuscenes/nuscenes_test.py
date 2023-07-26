"""Test NuScenes multi-sensor video dataset config."""
from __future__ import annotations

import unittest

from ml_collections import ConfigDict

from tests.util import get_test_data
from vis4d.config import instantiate_classes
from vis4d.config.common.datasets.nuscenes import (
    get_nusc_mini_train_cfg,
    get_nusc_mini_val_cfg,
)
from vis4d.data.datasets.nuscenes import NuScenes


class TestNuscenesConfig(unittest.TestCase):
    """Test cases for the nuscenes dataset configs."""

    NUSC_DATA_ROOT = get_test_data("nuscenes_test")

    def test_mini_train_cfg(self):
        """Test nuscenes mini train dataset config."""
        dataset_cfg = get_nusc_mini_train_cfg(data_root=self.NUSC_DATA_ROOT)
        self.assertTrue(isinstance(dataset_cfg, ConfigDict))
        nusc = instantiate_classes(dataset_cfg)
        self.assertTrue(isinstance(nusc, NuScenes))
        self.assertEqual(nusc.data_root, self.NUSC_DATA_ROOT)
        # Make sure it is callable. I.e. does not crash
        _ = next(iter(nusc))

    def test_mini_val_cfg(self):
        """Test nuscenes mini val dataset config."""
        dataset_cfg = get_nusc_mini_val_cfg(data_root=self.NUSC_DATA_ROOT)
        self.assertTrue(isinstance(dataset_cfg, ConfigDict))
        nusc = instantiate_classes(dataset_cfg)
        self.assertTrue(isinstance(nusc, NuScenes))
        self.assertEqual(nusc.data_root, self.NUSC_DATA_ROOT)
        # Make sure it is callable. I.e. does not crash
        _ = next(iter(nusc))
