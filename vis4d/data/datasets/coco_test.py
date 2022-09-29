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
            "metadata",
            "images",
            "boxes2d",
            "boxes2d_classes",
        )
