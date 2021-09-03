"""Testcases for VisT ScalabelDataset."""
import unittest
from typing import List

import torch
from scalabel.label.typing import Category, Config, Dataset, Frame

from ..struct import Images, InputSample
from .dataset import ScalabelDataset
from .datasets.base import (
    BaseDatasetConfig,
    BaseDatasetLoader,
    DataloaderConfig,
    ReferenceSamplingConfig,
)


class MockDatasetLoader(BaseDatasetLoader):
    """Scalabel dataset mockup."""

    def __init__(self, cfg: BaseDatasetConfig, frames: List[Frame]) -> None:
        """Init."""
        self.frames = frames
        super().__init__(cfg)

    def load_dataset(self) -> Dataset:
        """Load and possibly convert dataset to scalabel format."""
        config = Config(categories=[Category(name="test")])
        return Dataset(frames=self.frames, config=config)


class TestScalabelDataset(unittest.TestCase):
    """ScalabelDataset Testcase class."""

    cfg = BaseDatasetConfig(
        name="test",
        type="Scalabel",
        data_root="/path/to/root",
        dataloader=DataloaderConfig(
            ref_sampling=ReferenceSamplingConfig(
                type="sequential",
                num_ref_imgs=2,
                scope=3,
                frame_order="temporal",
            )
        ),
    )

    dataset_loader = MockDatasetLoader(
        cfg,
        [
            Frame(
                name=str(i),
                videoName=str(i % 2),
                frameIndex=i - i // 2 - i % 2,
            )
            for i in range(200)
        ],
    )

    dataset = ScalabelDataset(dataset_loader, True)

    def test_reference_sampling(self) -> None:
        """Testcase for reference view sampling."""
        idcs = self.dataset.sample_ref_indices(str(0), 50)
        self.assertTrue(idcs == [52, 54])
        idcs = self.dataset.sample_ref_indices(str(0), 196)
        self.assertTrue(idcs == [194, 198])

    def test_getitem_fallback(self) -> None:
        """Testcase for getitem fallback if None is returned."""
        cfg = BaseDatasetConfig(
            name="test",
            type="Scalabel",
            data_root="vist/engine/testcases/track/bdd100k-samples/images/",
            dataloader=DataloaderConfig(
                ref_sampling=ReferenceSamplingConfig(
                    type="sequential",
                    num_ref_imgs=1,
                    scope=3,
                    skip_nomatch_samples=True,
                ),
            ),
        )

        dataset_loader = MockDatasetLoader(
            cfg,
            [
                Frame(
                    name="00091078-875c1f73-0000167.jpg",
                    videoName="00091078-875c1f73",
                    frameIndex=i,
                )
                for i in range(6)
            ],
        )
        dataset = ScalabelDataset(dataset_loader, True)
        self.assertRaises(ValueError, dataset.__getitem__, 0)

    def test_transform_annotations(self) -> None:
        """Test the transform annotations method in DatasetMapper."""
        input_sample = InputSample(
            Frame(name="0"),
            Images(torch.zeros(1, 3, 128, 128), [(128, 128)]),
        )
        self.dataset.transform_annotation(input_sample, None, lambda x: x)
        self.assertEqual(len(input_sample.boxes2d), 0)
        self.dataset.transform_annotation(input_sample, [], lambda x: x)
        self.assertEqual(len(input_sample.boxes2d), 0)

    def test_sort_samples(self) -> None:
        """Test the sort_samples method in MapDataset."""
        input_samples = [
            InputSample(
                Frame(name="1", frameIndex=1),
                Images(torch.zeros(1, 3, 128, 128), [(128, 128)]),
            ),
            InputSample(
                Frame(name="0", frameIndex=0),
                Images(torch.zeros(1, 3, 128, 128), [(128, 128)]),
            ),
        ]
        sorted_samples = self.dataset.sort_samples(input_samples)
        self.assertEqual(sorted_samples[0].metadata.frameIndex, 0)
        self.assertEqual(sorted_samples[1].metadata.frameIndex, 1)
