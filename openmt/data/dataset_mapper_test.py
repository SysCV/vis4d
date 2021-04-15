"""Testcases for tracking dataset mapper."""
import unittest

from detectron2.data import DatasetFromList

from .dataset_mapper import MapTrackingDataset, ReferenceSamplingConfig


class TestDatasetMapper(unittest.TestCase):
    """DatasetMapper Testcase class."""

    def test_reference_sampling(self) -> None:
        """Testcase for reference view sampling."""
        cfg = ReferenceSamplingConfig(
            type="sequential", num_ref_imgs=2, scope=3
        )

        data_dict = [
            dict(video_id=str(i % 2), frame_id=i - i // 2 - i % 2)
            for i in range(200)
        ]
        mapper = MapTrackingDataset(
            cfg, DatasetFromList(data_dict), lambda x: x
        )
        video_idcs = mapper.video_to_idcs[str(0)]

        idcs = mapper.sample_ref_idcs(video_idcs, 50)
        self.assertTrue(idcs == [52, 54])

        idcs = mapper.sample_ref_idcs(video_idcs, 196)
        self.assertTrue(idcs == [194, 198])

    def test_getitem_fallback(self) -> None:
        """Testcase for getitem fallback if None is returned."""
        cfg = ReferenceSamplingConfig(
            type="sequential", num_ref_imgs=2, scope=3
        )

        data_dict = [dict(video_id=i % 2, frame_id=i) for i in range(200)]
        mapper = MapTrackingDataset(
            cfg, DatasetFromList(data_dict), lambda x: None
        )
        self.assertRaises(ValueError, mapper.__getitem__, 0)
