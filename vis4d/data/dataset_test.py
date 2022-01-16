"""Testcases for Vis4D ScalabelDataset."""
import unittest
from typing import List

import torch
from scalabel.label.typing import (
    Box2D,
    Category,
    Config,
    Dataset,
    Frame,
    Label,
)

from ..struct import ArgsType, Boxes2D, Images, InputSample
from .dataset import ScalabelDataset
from .datasets import Scalabel


class MockDatasetLoader(Scalabel):
    """Scalabel dataset mockup."""

    def __init__(
        self, *args: ArgsType, frames: List[Frame], **kwargs: ArgsType
    ) -> None:
        """Init."""
        self.frames = frames
        super().__init__(*args, **kwargs)

    def load_dataset(self) -> Dataset:
        """Load and possibly convert dataset to scalabel format."""
        config = Config(categories=[Category(name="test")])
        return Dataset(frames=self.frames, config=config)


class TestScalabelDataset(unittest.TestCase):
    """ScalabelDataset Testcase class."""

    dataset_loader = MockDatasetLoader(
        frames=[
            Frame(
                name=str(i),
                videoName=str(i % 2),
                frameIndex=i - i // 2 - i % 2,
            )
            for i in range(200)
        ],
        name="mock_dataset",
        data_root="/path/to/root",
    )

    dataset = ScalabelDataset(
        dataset_loader,
        mapper_cfg={},
        ref_cfg={
            "strategy": "sequential",
            "num_ref_imgs": 2,
            "scope": 3,
            "frame_order": "temporal",
        },
        training=True,
    )

    def test_reference_sampling(self) -> None:
        """Testcase for reference view sampling."""
        idcs = self.dataset.ref_sampler.sample_ref_indices(str(0), 50)
        self.assertTrue(idcs == [52, 54])
        idcs = self.dataset.ref_sampler.sample_ref_indices(str(0), 196)
        self.assertTrue(idcs == [194, 198])

    def test_getitem_fallback(self) -> None:
        """Testcase for getitem fallback if None is returned."""
        dataset_loader = MockDatasetLoader(
            frames=[
                Frame(
                    name="00091078-875c1f73-0000167.jpg",
                    videoName="00091078-875c1f73",
                    frameIndex=i,
                )
                for i in range(6)
            ],
            name="mock_dataset",
            data_root="vis4d/engine/testcases/track/bdd100k-samples/images/",
        )
        dataset = ScalabelDataset(
            dataset_loader,
            mapper_cfg={},
            ref_cfg={
                "strategy": "sequential",
                "num_ref_imgs": 1,
                "scope": 3,
                "skip_nomatch_samples": True,
            },
            training=True,
        )
        # assert makes sure that all samples will be discarded from fallback
        # candidates (due to no match) and subsequently raises a ValueError
        # since there is no fallback candidates to sample from anymore
        self.assertRaises(ValueError, dataset.__getitem__, 0)

    def test_transform_input(self) -> None:
        """Test the transform_input method in ScalabelDataset."""
        sample = InputSample(
            [Frame(name="0")],
            Images(torch.zeros(1, 3, 128, 128), [(128, 128)]),
        )
        self.dataset.mapper.transform_input(sample, None)
        self.assertEqual(len(sample.targets.boxes2d[0]), 0)
        self.dataset.mapper.transform_input(sample, [])
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
        self.dataset.mapper.transform_input(sample, [])

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

    def test_filter_attributes(self) -> None:
        """Testcase for attribute filtering."""
        # Testcase 1
        dataset_loader = MockDatasetLoader(
            frames=[
                Frame(
                    name=str(i),
                    videoName=str(i % 2),
                    frameIndex=i - i // 2 - i % 2,
                    attributes={"timeofday": "daytime", "weather": "clear"},
                )
                for i in range(6)
            ],
            name="mock_dataset",
            data_root="/path/to/root",
            attributes={"timeofday": ["daytime", "night"], "weather": "clear"},
        )

        ref_cfg = {
            "strategy": "sequential",
            "num_ref_imgs": 2,
            "scope": 3,
            "frame_order": "temporal",
        }
        dataset = ScalabelDataset(
            dataset_loader, mapper_cfg={}, ref_cfg=ref_cfg, training=True
        )
        self.assertTrue(len(dataset) == 6)

        # Testcase 2
        dataset_loader.frames = [
            Frame(
                name=str(i),
                videoName=str(i % 2),
                frameIndex=i - i // 2 - i % 2,
                attributes={"timeofday": "night", "weather": "clear"},
            )
            for i in range(6)
        ]

        dataset = ScalabelDataset(
            dataset_loader, mapper_cfg={}, ref_cfg=ref_cfg, training=True
        )
        self.assertTrue(len(dataset) == 6)

        # Testcase 3
        dataset_loader.frames = [
            Frame(
                name=str(i),
                videoName=str(i % 2),
                frameIndex=i - i // 2 - i % 2,
                attributes={"timeofday": "daytime", "weather": "snowy"},
            )
            for i in range(6)
        ]

        self.assertRaises(
            ValueError, ScalabelDataset, dataset_loader, {}, ref_cfg, True
        )
