"""COCO dataset testing class."""
import unittest

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as Keys
from vis4d.data.datasets.coco import COCO


class COCOTest(unittest.TestCase):
    """Test coco dataloading."""

    coco = COCO(
        data_root=get_test_data("coco_test"),
        split="train",
        keys_to_load=(
            Keys.images,
            Keys.boxes2d,
            Keys.boxes2d_classes,
            Keys.masks,
        ),
    )

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.coco), 2)

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        item = self.coco[0]
        self.assertEqual(
            tuple(item.keys()),
            (
                "original_hw",
                "input_hw",
                "coco_image_id",
                "images",
                "boxes2d",
                "boxes2d_classes",
                "masks",
            ),
        )
        self.assertEqual(len(item[Keys.boxes2d]), 14)
        self.assertEqual(len(item[Keys.boxes2d_classes]), 14)
        self.assertEqual(len(item[Keys.masks]), 14)


class COCOSegTest(unittest.TestCase):
    """Test coco dataloading."""

    coco = COCO(
        data_root=get_test_data("coco_test"),
        split="train",
        keys_to_load=(
            Keys.images,
            Keys.boxes2d,
            Keys.boxes2d_classes,
            Keys.masks,
            Keys.segmentation_masks,
        ),
        remove_empty=True,
        minimum_box_area=10,
        use_pascal_voc_cats=True,
    )

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.coco), 2)

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        item = self.coco[0]
        assert tuple(item.keys()) == (
            "original_hw",
            "input_hw",
            "coco_image_id",
            "images",
            "boxes2d",
            "boxes2d_classes",
            "masks",
            "segmentation_masks",
        )
        self.assertEqual(item[Keys.segmentation_masks].shape, (1, 230, 352))
