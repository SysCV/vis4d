"""Test cases for dataset and dataloader using s3dis."""
from __future__ import annotations

import unittest

import torch

from tests.util import get_test_data
from vis4d.config import instantiate_classes
from vis4d.config.common.datasets.s3dis.sem_seg import get_s3dis_sem_seg_cfg
from vis4d.data.const import CommonKeys as K


class TestDataloaderConfig(unittest.TestCase):
    """Test cases for the dataloader config."""

    DATA_ROOT = get_test_data("s3d_test")

    def test_dataloader_config(self) -> None:
        """Test case to instantiate a dataloader from a config."""
        dataloader_cfg = get_s3dis_sem_seg_cfg(data_root=self.DATA_ROOT)
        dl = instantiate_classes(dataloader_cfg)
        entry = next(iter(dl.train_dataloader))

        self.assertEqual(entry[K.points3d][0].shape, (24000, 3))
        self.assertEqual(entry[K.semantics3d][0].shape, torch.Size([24000]))
        self.assertEqual(entry[K.colors3d][0].shape, (24000, 3))

        entry = next(iter(dl.test_dataloader[0]))
        self.assertEqual(entry[K.colors3d][0].shape, (39041, 3))
