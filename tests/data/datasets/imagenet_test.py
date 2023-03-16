"""ImageNet 1K dataset tests."""

import unittest

import torch

from tests.util import get_test_data, isclose_on_all_indices
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.imagenet import ImageNet

IMAGE_INDICES = torch.tensor([0, 1, 113250, 226499])
IMAGE_VALUES = torch.tensor(
    [
        [54.0, 93.0, 148.0],
        [52.0, 92.0, 144.0],
        [37.0, 87.0, 148.0],
        [28.0, 31.0, 38.0],
    ]
)


class ImageNetTest(unittest.TestCase):
    """Test ImageNet 1K dataloading."""

    dataset = ImageNet(
        data_root=get_test_data("imagenet_1k_test"),
        split="train",
        keys_to_load=(
            K.images,
            K.categories,
        ),
        num_classes=2,
    )

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.dataset), 3)

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        self.assertEqual(
            tuple(self.dataset[0].keys()), ("images", "categories")
        )
        self.assertEqual(self.dataset[0][K.categories], 0)
        self.assertEqual(self.dataset[1][K.categories], 1)
        self.assertEqual(self.dataset[2][K.categories], 1)
        self.assertEqual(
            self.dataset[0][K.images].shape, torch.Size([1, 3, 500, 453])
        )
        assert isclose_on_all_indices(
            self.dataset[0][K.images].permute(0, 2, 3, 1).reshape(-1, 3),
            IMAGE_INDICES,
            IMAGE_VALUES,
        )
