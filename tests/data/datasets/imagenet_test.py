"""ImageNet 1K dataset tests."""

import unittest

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as Keys
from vis4d.data.datasets.imagenet import ImageNet


class ImageNetTest(unittest.TestCase):
    """Test ImageNet 1K dataloading."""

    dataset = ImageNet(
        data_root=get_test_data("imagenet_1k_test"),
        split="train",
        keys_to_load=(
            Keys.images,
            Keys.categories,
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
        self.assertEqual(self.dataset[0][Keys.categories], 0)
