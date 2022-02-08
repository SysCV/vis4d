"""Testcases for Vis4D DataModule."""
import unittest

import torch
from scalabel.label.typing import Box2D, Frame, Label

from ..struct import Boxes2D, Images, InputSample
from .dataset_test import TestScalabelDataset


class TestDataModule(unittest.TestCase):
    """ScalabelDataset Testcase class."""

    dataset = TestScalabelDataset.dataset

    def test_transform_input(self) -> None:
        """Test the transform_input method in ScalabelDataset."""
        sample = InputSample(
            [Frame(name="0")],
            Images(torch.zeros(1, 3, 128, 128), [(128, 128)]),
        )
        self.handler.transform_inputs(sample, None)
        self.assertEqual(len(sample.targets.boxes2d[0]), 0)
        self.dataset.mapper.transform_inputs(sample, [])
        self.assertEqual(len(sample.targets.boxes2d[0]), 0)

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
        self.dataset.mapper.transform_inputs(sample, [])

        self.assertTrue(all(sample.targets.boxes2d[0].class_ids == 0))
        self.assertEqual(sample.targets.boxes2d[0].boxes[0, 0], 10)
        self.assertEqual(sample.targets.boxes2d[0].boxes[1, 0], 11)
        self.assertEqual(sample.targets.boxes2d[0].boxes[2, 0], 12)

        self.assertEqual(sample.targets.boxes2d[0].track_ids[0], 2)
        self.assertEqual(sample.targets.boxes2d[0].track_ids[1], 1)
        self.assertEqual(sample.targets.boxes2d[0].track_ids[2], 0)

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
        sorted_samples = self.dataset.ref_sampler.sort_samples(input_samples)
        self.assertEqual(sorted_samples[0].metadata[0].frameIndex, 0)
        self.assertEqual(sorted_samples[1].metadata[0].frameIndex, 1)
