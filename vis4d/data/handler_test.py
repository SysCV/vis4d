"""Testcases for Vis4D DataModule."""
import unittest

import torch
from scalabel.label.typing import Box2D, Frame, Label

from ..struct import Boxes2D, Images, InputSample
from .dataset_test import TestScalabelDataset
from .handler import Vis4DDatasetHandler, sort_by_frame_index


class TestDataHandler(unittest.TestCase):
    """DataHandler Testcase class."""

    dataset = TestScalabelDataset.dataset
    handler = Vis4DDatasetHandler([dataset])

    @staticmethod
    def _make_test_sample() -> InputSample:
        """Create new sample mockup for testing."""
        sample = InputSample(
            [Frame(name="0")],
            Images(torch.zeros(1, 3, 128, 128), [(128, 128)]),
        )
        labels = [
            Label(
                id="a",
                category="car",
                box2d=Box2D(x1=10, y1=10, x2=20, y2=20),
                attributes={"category_id": 0, "instance_id": 2},
            ),
            Label(
                id="b",
                category="car",
                box2d=Box2D(x1=11, y1=10, x2=20, y2=20),
                attributes={"category_id": 0, "instance_id": 1},
            ),
            Label(
                id="c",
                category="car",
                box2d=Box2D(x1=12, y1=10, x2=20, y2=20),
                attributes={"category_id": 0, "instance_id": 0},
            ),
        ]
        sample.targets.boxes2d = [
            Boxes2D.from_scalabel(labels, {"car": 0}, {"a": 2, "b": 1, "c": 0})
        ]
        return sample

    def test_postprocess_annotations(self) -> None:
        """Test postprocessing of annotations."""
        sample = self._make_test_sample()
        self.handler.min_bboxes_area = 12 * 12
        # pylint: disable=protected-access
        self.handler._postprocess_annotations((128, 128), sample.targets)
        self.assertEqual(len(sample.targets.boxes2d[0]), 0)
        self.handler.min_bboxes_area = 7 * 7

    def test_rescale_track_ids(self) -> None:
        """Test rescaling of track ids."""
        sample = self._make_test_sample()
        sample.targets.boxes2d[0].track_ids = torch.tensor([10, 20, 30])
        # pylint: disable=protected-access
        self.handler._rescale_track_ids([sample])
        self.assertEqual(
            tuple(sample.targets.boxes2d[0].track_ids.numpy()), (0, 1, 2)
        )

    def test_sort_samples(self) -> None:
        """Test the sort_samples method in MapDataset."""
        input_samples = [
            InputSample(
                [Frame(name="1", frameIndex=1)],
                Images(torch.zeros(1, 3, 128, 128), [(128, 128)]),
            ),
            InputSample(
                [Frame(name="0", frameIndex=0)],
                Images(torch.zeros(1, 3, 128, 128), [(128, 128)]),
            ),
        ]
        sorted_samples = sort_by_frame_index(input_samples)
        self.assertEqual(sorted_samples[0].metadata[0].frameIndex, 0)
        self.assertEqual(sorted_samples[1].metadata[0].frameIndex, 1)
