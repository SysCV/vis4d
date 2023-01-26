"""COCO dataset testing class."""
import unittest

from tests.util import get_test_data
from vis4d.data.datasets.coco import COCO


class COCOTest(unittest.TestCase):
    """Test coco dataloading."""

    coco = COCO(data_root=get_test_data("coco_test"), split="train")

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.coco), 2)

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        assert tuple(self.coco[0].keys()) == (
            "original_hw",
            "input_hw",
            "coco_image_id",
            "images",
            "boxes2d",
            "boxes2d_classes",
            "masks",
        )


class COCOSegTest(unittest.TestCase):
    """Test coco dataloading."""

    coco = COCO(
        data_root=get_test_data("coco_test"),
        split="train",
        remove_empty=True,
        minimum_box_area=1000,
        use_pascal_voc_cats=True,
    )

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.coco), 2)

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        assert tuple(self.coco[0].keys()) == (
            "original_hw",
            "input_hw",
            "coco_image_id",
            "images",
            "boxes2d",
            "boxes2d_classes",
            "masks",
        )
