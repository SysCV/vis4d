"""COCO dataset testing class."""
import unittest

from .coco import COCO


class COCOTest(unittest.TestCase):
    """Test coco dataloading."""

    coco = COCO(data_root="data/COCO/")

    def test_len(self):
        """Test if len of dataset correct."""
        self.assertEqual(len(self.coco), 118287)

    def test_sample(self):
        """Test if sample loaded correctly."""
        assert tuple(self.coco[0].keys()) == (
            "original_hw",
            "input_hw",
            "transform_params",
            "batch_transform_params",
            "coco_image_id",
            "images",
            "boxes2d",
            "boxes2d_classes",
            "masks",
        )


class COCOSegTest(unittest.TestCase):
    """Test coco dataloading."""

    coco = COCO(
        data_root="data/COCO/",
        remove_empty=True,
        minimum_box_area=1000,
        use_pascal_voc_cats=True,
    )

    def test_len(self):
        """Test if len of dataset correct."""
        self.assertEqual(len(self.coco), 92518)

    def test_sample(self):
        """Test if sample loaded correctly."""
        assert tuple(self.coco[0].keys()) == (
            "original_hw",
            "input_hw",
            "transform_params",
            "batch_transform_params",
            "coco_image_id",
            "images",
            "boxes2d",
            "boxes2d_classes",
            "masks",
        )
