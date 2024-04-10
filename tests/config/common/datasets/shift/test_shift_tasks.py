"""Test cases for dataset and dataloader using COCO."""

from __future__ import annotations

import unittest

from ml_collections import ConfigDict
from torch.utils.data.dataloader import DataLoader

from tests.util import get_test_data
from vis4d.config import class_config, instantiate_classes
from vis4d.config.common.datasets.shift import (
    get_shift_depth_est_config,
    get_shift_det_config,
    get_shift_instance_seg_config,
    get_shift_multitask_2d_config,
    get_shift_sem_seg_config,
)
from vis4d.data.const import CommonKeys as K
from vis4d.data.io import HDF5Backend


class TestMultiViewDataloaderConfig(unittest.TestCase):
    """Test cases for the dataloader config for multiview dataset."""

    DATA_ROOT = get_test_data("shift_test")
    VIEWS = ["front"]
    SPLIT = "val"
    DOMAIN_ATTR = [{"weather_coarse": "rainy", "timeofday_coarse": "night"}]

    def test_sem_seg_config(self) -> None:
        """Test case to instantiate a dataloader from a config.

        This also checks that the detection preprocessing works.
        """
        dataloader_cfg = get_shift_sem_seg_config(
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
        self.assertEqual(entries[K.images].shape, (1, 3, 512, 1024))
        self.assertEqual(entries[K.seg_masks].shape, (1, 512, 1024))
        self.assertEqual(entries[K.original_hw], [(800, 1280)])

        test_dl = instantiate_classes(dataloader_cfg.test_dataloader)
        self.assertTrue(isinstance(test_dl[0], DataLoader))
        entries = next(iter(test_dl[0]))
        self.assertEqual(entries[K.images].shape, (1, 3, 800, 1280))
        self.assertEqual(entries[K.seg_masks].shape, (1, 800, 1280))
        self.assertEqual(entries[K.original_hw], [(800, 1280)])

    def test_det_config(self) -> None:
        """Test case to instantiate a dataloader from a config.

        This also checks that the preprocessing works.
        """
        dataloader_cfg = get_shift_det_config(
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
        self.assertEqual(entries[K.images].shape, (1, 3, 800, 1280))
        self.assertEqual(entries[K.original_hw], [(800, 1280)])
        self.assertEqual(entries[K.boxes2d][0].shape, (2, 4))
        self.assertEqual(entries[K.boxes2d_classes][0].shape, (2,))

        test_dl = instantiate_classes(dataloader_cfg.test_dataloader)
        self.assertTrue(isinstance(test_dl[0], DataLoader))
        entries = next(iter(test_dl[0]))
        self.assertEqual(entries[K.images].shape, (1, 3, 800, 1280))
        self.assertEqual(entries[K.original_hw], [(800, 1280)])
        self.assertEqual(entries[K.boxes2d][0].shape, (2, 4))
        self.assertEqual(entries[K.boxes2d_classes][0].shape, (2,))

    def test_depth_config(self) -> None:
        """Test case to instantiate a dataloader from a config.

        This also checks that the preprocessing works.
        """
        dataloader_cfg = get_shift_depth_est_config(
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
        self.assertEqual(entries[K.images].shape, (1, 3, 800, 1280))
        self.assertEqual(entries[K.original_hw], [(800, 1280)])
        self.assertEqual(entries[K.depth_maps].shape, (1, 800, 1280))

        test_dl = instantiate_classes(dataloader_cfg.test_dataloader)
        self.assertTrue(isinstance(test_dl[0], DataLoader))
        entries = next(iter(test_dl[0]))
        self.assertEqual(entries[K.images].shape, (1, 3, 800, 1280))
        self.assertEqual(entries[K.original_hw], [(800, 1280)])
        self.assertEqual(entries[K.depth_maps].shape, (1, 800, 1280))

    def test_instance_seg_config(self) -> None:
        """Test case to instantiate a dataloader from a config.

        This also checks that the preprocessing works.
        """
        dataloader_cfg = get_shift_instance_seg_config(
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
        self.assertEqual(entries[K.images].shape, (1, 3, 800, 1280))
        self.assertEqual(entries[K.original_hw], [(800, 1280)])
        self.assertEqual(entries[K.boxes2d][0].shape, (2, 4))
        self.assertEqual(entries[K.boxes2d_classes][0].shape, (2,))
        self.assertEqual(entries[K.instance_masks][0].shape, (2, 800, 1280))

        test_dl = instantiate_classes(dataloader_cfg.test_dataloader)
        self.assertTrue(isinstance(test_dl[0], DataLoader))
        entries = next(iter(test_dl[0]))
        self.assertEqual(entries[K.images].shape, (1, 3, 800, 1280))
        self.assertEqual(entries[K.original_hw], [(800, 1280)])
        self.assertEqual(entries[K.boxes2d][0].shape, (2, 4))
        self.assertEqual(entries[K.boxes2d_classes][0].shape, (2,))
        self.assertEqual(entries[K.instance_masks][0].shape, (2, 800, 1280))

    def test_multitask_config(self) -> None:
        """Test case to instantiate a dataloader from a config.

        This also checks that the preprocessing works.
        """
        dataloader_cfg = get_shift_multitask_2d_config(
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
