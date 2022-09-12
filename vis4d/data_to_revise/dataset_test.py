"""Testcases for Vis4D Dataset."""
import unittest
from typing import List

from scalabel.label.typing import Category, Config, Dataset, Frame

from ..struct_to_revise import ArgsType
from .dataset import ScalabelDataset
from .datasets import Scalabel
from .mapper import BaseSampleMapper
from .reference import BaseReferenceSampler


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
        training=True,
        ref_sampler=BaseReferenceSampler(
            strategy="sequential",
            num_ref_imgs=2,
            scope=3,
            frame_order="temporal",
        ),
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
            training=True,
            mapper=BaseSampleMapper(category_map={"test": 0}),
            ref_sampler=BaseReferenceSampler(
                strategy="sequential",
                num_ref_imgs=1,
                scope=3,
                skip_nomatch_samples=True,
            ),
        )
        # assert makes sure that all samples will be discarded from fallback
        # candidates (due to no match) and subsequently raises a ValueError
        # since there is no fallback candidates to sample from anymore
        self.assertRaises(ValueError, dataset.__getitem__, 0)

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

        ref_sampler = BaseReferenceSampler(
            strategy="sequential",
            num_ref_imgs=2,
            scope=3,
            frame_order="temporal",
        )
        dataset = ScalabelDataset(
            dataset_loader, training=True, ref_sampler=ref_sampler
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

        ref_sampler.frames = None
        dataset = ScalabelDataset(
            dataset_loader, training=True, ref_sampler=ref_sampler
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
            ValueError,
            ScalabelDataset,
            dataset_loader,
            True,
            None,
            ref_sampler,
        )
