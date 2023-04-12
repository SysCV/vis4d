"""BDD100K dataset testing class."""
import os
import unittest

import torch

from tests.util import get_test_data, isclose_on_all_indices_tensor
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.bdd100k import BDD100K
from vis4d.data.transforms.to_tensor import ToTensor

IMAGE_INDICES = torch.tensor([0, 1, 460800, 921599])
IMAGE_VALUES = torch.tensor(
    [
        [167.0, 192.0, 197.0],
        [167.0, 192.0, 197.0],
        [12.0, 14.0, 11.0],
        [20.0, 21.0, 23.0],
    ]
)
INSTANCE_MASK_INDICES = torch.tensor([0, 1, 406208, 1382400, 3173655])
INSTANCE_MASK_VALUES = torch.tensor([0, 0, 1, 0, 1]).byte()
SEMANTIC_MASK_INDICES = torch.tensor([10, 1000, 10000, 500000])
SEMANTIC_MASK_VALUES = torch.tensor([10, 255, 2, 0]).byte()


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
            K.images,
            K.boxes2d,
            K.boxes2d_classes,
            K.boxes2d_track_ids,
        ),
        config_path=config_path,
    )

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.dataset), 1)

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        item = self.dataset[0]
        item = ToTensor().apply_to_data([item])[0]  # pylint: disable=no-member
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

        self.assertEqual(item[K.images].shape, (1, 3, 720, 1280))
        self.assertEqual(len(item[K.boxes2d]), 10)
        self.assertEqual(len(item[K.boxes2d_classes]), 10)
        self.assertEqual(len(item[K.boxes2d_track_ids]), 10)

        self.assertEqual(item["original_hw"], (720, 1280))
        self.assertEqual(item["input_hw"], (720, 1280))
        self.assertEqual(item["name"], "913b47b8-3cf1b886.jpg")
        self.assertEqual(item["videoName"], None)

        assert isclose_on_all_indices_tensor(
            item[K.images].permute(0, 2, 3, 1).reshape(-1, 3),
            IMAGE_INDICES,
            IMAGE_VALUES,
        )
        assert torch.isclose(
            item[K.boxes2d][0],
            torch.tensor(
                [624.2538, 290.0232, 636.5168, 303.4125], dtype=torch.float32
            ),
        ).all()
        assert torch.isclose(
            item[K.boxes2d_classes],
            torch.tensor([8, 8, 8, 8, 8, 9, 2, 2, 2, 2], dtype=torch.long),
        ).all()
        assert torch.isclose(
            item[K.boxes2d_track_ids],
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long),
        ).all()

    def test_mapping(self):
        """Test if mapping is generated correctly."""
        data = (
            self.dataset._generate_mapping()  # pylint: disable=protected-access,line-too-long
        )
        assert len(data.frames) == 1
        assert len(data.config.categories) == 5
        assert len(data.frames[0].labels) == 10


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
            K.images,
            K.boxes2d,
            K.boxes2d_classes,
            K.boxes2d_track_ids,
            K.instance_masks,
        ),
        config_path=config_path,
        global_instance_ids=True,
    )

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.dataset), 1)

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        item = self.dataset[0]
        item = ToTensor().apply_to_data([item])[0]  # pylint: disable=no-member
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
                "instance_masks",
            ),
        )

        self.assertEqual(len(item[K.boxes2d]), 10)
        self.assertEqual(len(item[K.boxes2d_classes]), 10)
        self.assertEqual(len(item[K.boxes2d_track_ids]), 10)
        self.assertEqual(item[K.instance_masks].shape, (4, 720, 1280))

        self.assertEqual(item["original_hw"], (720, 1280))
        self.assertEqual(item["input_hw"], (720, 1280))
        self.assertEqual(item["name"], "913b47b8-3cf1b886.jpg")
        self.assertEqual(item["videoName"], None)

        assert isclose_on_all_indices_tensor(
            item[K.images].permute(0, 2, 3, 1).reshape(-1, 3),
            IMAGE_INDICES,
            IMAGE_VALUES,
        )
        assert isclose_on_all_indices_tensor(
            item[K.instance_masks].reshape(-1),
            INSTANCE_MASK_INDICES,
            INSTANCE_MASK_VALUES,
        )
        assert torch.isclose(
            item[K.boxes2d][0],
            torch.tensor(
                [624.2538, 290.0232, 636.5168, 303.4125], dtype=torch.float32
            ),
        ).all()
        assert torch.isclose(
            item[K.boxes2d_classes],
            torch.tensor([8, 8, 8, 8, 8, 9, 2, 2, 2, 2], dtype=torch.long),
        ).all()
        assert torch.isclose(
            item[K.boxes2d_track_ids],
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long),
        ).all()


class BDD100KSemSegTest(unittest.TestCase):
    """Test BDD100K dataloading."""

    bdd_root = get_test_data("bdd100k_test")
    data_root = os.path.join(bdd_root, "segment/images")
    annotations = os.path.join(bdd_root, "segment/labels/annotation.json")
    config_path = "sem_seg"

    dataset = BDD100K(
        data_root,
        annotations,
        keys_to_load=(K.images, K.seg_masks),
        config_path=config_path,
        global_instance_ids=True,
    )

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.dataset), 2)

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        item = self.dataset[0]
        item = ToTensor().apply_to_data([item])[0]  # pylint: disable=no-member
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
                "seg_masks",
            ),
        )

        self.assertEqual(item[K.seg_masks].shape, (720, 1280))
        self.assertEqual(item["original_hw"], (720, 1280))
        self.assertEqual(item["input_hw"], (720, 1280))
        self.assertEqual(item["name"], "913b47b8-3cf1b886.jpg")
        self.assertEqual(item["videoName"], None)

        assert isclose_on_all_indices_tensor(
            item[K.images].permute(0, 2, 3, 1).reshape(-1, 3),
            IMAGE_INDICES,
            IMAGE_VALUES,
        )
        assert isclose_on_all_indices_tensor(
            item[K.seg_masks].reshape(-1),
            SEMANTIC_MASK_INDICES,
            SEMANTIC_MASK_VALUES,
        )

    def test_bg_as_class(self):
        """Test if background class is added."""
        dataset = BDD100K(
            self.data_root,
            self.annotations,
            keys_to_load=(K.images, K.seg_masks),
            category_map={"car": 0, "person": 1, "background": 2},
            global_instance_ids=True,
            bg_as_class=True,
        )
        item = dataset[0]
        item = ToTensor().apply_to_data([item])[0]  # pylint: disable=no-member
        assert 2 in item["seg_masks"].unique()
        assert 255 not in item["seg_masks"].unique()
