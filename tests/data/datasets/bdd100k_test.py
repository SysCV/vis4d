"""BDD100K dataset testing class."""
import os
import unittest

import torch

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as Keys
from vis4d.data.datasets.bdd100k import BDD100K


class BDD100KDetTest(unittest.TestCase):
    """Test BDD100K dataloader."""

    bdd_root = get_test_data("bdd100k_test")
    data_root = os.path.join(bdd_root, "detect/images")
    annotations = os.path.join(bdd_root, "detect/labels/annotation.json")
    config_path = os.path.join(bdd_root, "detect/config.toml")

    dataset = BDD100K(
        data_root,
        annotations,
        keys_to_load=(
            Keys.images,
            Keys.boxes2d,
            Keys.boxes2d_classes,
            Keys.boxes2d_track_ids,
        ),
        config_path=config_path,
    )

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.dataset), 1)

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        item = self.dataset[0]
        self.assertEqual(
            tuple(item.keys()),
            (
                "images",
                "original_hw",
                "input_hw",
                "axis_mode",
                "frame_ids",
                "name",
                "videoName",
                "boxes2d",
                "boxes2d_classes",
                "boxes2d_track_ids",
            ),
        )

        self.assertEqual(item[Keys.images].shape, (1, 3, 720, 1280))
        self.assertEqual(len(item[Keys.boxes2d]), 10)
        self.assertEqual(len(item[Keys.boxes2d_classes]), 10)
        self.assertEqual(len(item[Keys.boxes2d_track_ids]), 10)

        assert torch.isclose(
            item[Keys.boxes2d_classes],
            torch.tensor([8, 8, 8, 8, 8, 9, 2, 2, 2, 2], dtype=torch.long),
        ).all()
        assert torch.isclose(
            item[Keys.boxes2d_track_ids],
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long),
        ).all()


class BDD100KInsSegTest(unittest.TestCase):
    """Test BDD100K dataloading."""

    bdd_root = get_test_data("bdd100k_test")
    data_root = os.path.join(bdd_root, "detect/images")
    annotations = os.path.join(bdd_root, "detect/labels/annotation.json")
    config_path = os.path.join(bdd_root, "detect/config.toml")

    dataset = BDD100K(
        data_root,
        annotations,
        keys_to_load=(
            Keys.images,
            Keys.boxes2d,
            Keys.boxes2d_classes,
            Keys.boxes2d_track_ids,
            Keys.masks,
        ),
        config_path=config_path,
    )

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.dataset), 1)

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        item = self.dataset[0]
        self.assertEqual(
            tuple(item.keys()),
            (
                "images",
                "original_hw",
                "input_hw",
                "axis_mode",
                "frame_ids",
                "name",
                "videoName",
                "boxes2d",
                "boxes2d_classes",
                "boxes2d_track_ids",
                "masks",
            ),
        )

        self.assertEqual(len(item[Keys.boxes2d]), 10)
        self.assertEqual(len(item[Keys.boxes2d_classes]), 10)
        self.assertEqual(len(item[Keys.boxes2d_track_ids]), 10)
        self.assertEqual(item[Keys.masks].shape, (4, 720, 1280))
