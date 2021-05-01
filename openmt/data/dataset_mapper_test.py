"""Testcases for tracking dataset mapper."""
import unittest

import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetFromList
from scalabel.label.typing import Frame

from openmt.common.io import DataBackendConfig
from openmt.struct import Images, InputSample

from .dataset_mapper import DatasetMapper, MapDataset, ReferenceSamplingConfig


class TestDatasetMapper(unittest.TestCase):
    """DatasetMapper Testcase class."""

    def test_reference_sampling(self) -> None:
        """Testcase for reference view sampling."""
        cfg = ReferenceSamplingConfig(
            type="sequential", num_ref_imgs=2, scope=3
        )

        data_dict = [
            dict(video_name=str(i % 2), frame_index=i - i // 2 - i % 2)
            for i in range(200)
        ]
        mapper = MapDataset(cfg, True, DatasetFromList(data_dict), lambda x: x)

        idcs = mapper.sample_ref_idcs(str(0), 50)
        self.assertTrue(idcs == [52, 54])

        idcs = mapper.sample_ref_idcs(str(0), 196)
        self.assertTrue(idcs == [194, 198])

    def test_getitem_fallback(self) -> None:
        """Testcase for getitem fallback if None is returned."""
        cfg = ReferenceSamplingConfig(
            type="sequential", num_ref_imgs=2, scope=3
        )

        data_dict = [
            dict(video_name=i % 2, frame_index=i - i // 2 - i % 2)
            for i in range(200)
        ]
        mapper = MapDataset(
            cfg, True, DatasetFromList(data_dict), lambda x: None
        )
        self.assertRaises(ValueError, mapper.__getitem__, 0)

    def test_transform_annotations(self) -> None:
        """Test the transform annotations method in TrackingDatasetMapper."""
        cfg = get_cfg()
        ds_mapper = DatasetMapper(DataBackendConfig(), cfg)
        input_dict = InputSample(
            Frame(name="0"),
            Images(torch.zeros(1, 3, 128, 128), [(128, 128)]),
        )
        boxs = ds_mapper.transform_annotation(input_dict, None, lambda x: x)
        self.assertEqual(len(boxs), 0)
        boxs = ds_mapper.transform_annotation(input_dict, [], lambda x: x)
        self.assertEqual(len(boxs), 0)
