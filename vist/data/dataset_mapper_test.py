"""Testcases for VisT dataset mapper."""
import unittest

import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetFromList
from scalabel.label.typing import Frame

from vist.struct import Images, InputSample

from .dataset_mapper import (
    DataloaderConfig,
    DatasetMapper,
    MapDataset,
    ReferenceSamplingConfig,
)


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
        """Test the transform annotations method in DatasetMapper."""
        cfg = get_cfg()
        loader_cfg = DataloaderConfig(
            workers_per_gpu=0,
            image_channel_mode="BGR",
            ref_sampling_cfg=ReferenceSamplingConfig(num_ref_imgs=1, scope=1),
        )
        ds_mapper = DatasetMapper(loader_cfg, cfg)
        input_dict = InputSample(
            Frame(name="0"),
            Images(torch.zeros(1, 3, 128, 128), [(128, 128)]),
        )
        boxs = ds_mapper.transform_annotation(input_dict, None, lambda x: x)
        self.assertEqual(len(boxs), 0)
        boxs = ds_mapper.transform_annotation(input_dict, [], lambda x: x)
        self.assertEqual(len(boxs), 0)

    def test_sort_samples(self) -> None:
        """Test the sort_samples method in MapDataset."""
        cfg = ReferenceSamplingConfig(
            num_ref_imgs=1, scope=1, frame_order="temporal"
        )
        data_dict = [
            dict(video_name=i % 2, frame_index=i - i // 2 - i % 2)
            for i in range(200)
        ]
        ds_mapper = MapDataset(
            cfg, True, DatasetFromList(data_dict), lambda x: None
        )
        input_samples = [
            InputSample(
                Frame(name="1", frame_index=1),
                Images(torch.zeros(1, 3, 128, 128), [(128, 128)]),
            ),
            InputSample(
                Frame(name="0", frame_index=0),
                Images(torch.zeros(1, 3, 128, 128), [(128, 128)]),
            ),
        ]

        sorted_samples = ds_mapper.sort_samples(input_samples)
        self.assertEqual(sorted_samples[0].metadata.frame_index, 0)
        self.assertEqual(sorted_samples[1].metadata.frame_index, 1)
