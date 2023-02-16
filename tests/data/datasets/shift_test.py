"""S3DIS dataset testing class."""
import unittest

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as Keys
from vis4d.data.datasets.shift import SHIFT
from vis4d.data.io import HDF5Backend, ZipBackend


class SHIFTTest(unittest.TestCase):
    """Test S3DIS dataloading."""

    dataset = SHIFT(data_root=get_test_data("shift_test"), split="val")

    dataset_zip = SHIFT(
        data_root=get_test_data("shift_test"),
        split="val",
        keys_to_load=[
            Keys.images,
            Keys.input_hw,
            Keys.intrinsics,
            Keys.boxes2d,
        ],
        backend=ZipBackend(),
    )

    dataset_multiview = SHIFT(
        data_root=get_test_data("shift_test"),
        split="val",
        keys_to_load=[
            Keys.images,
            Keys.input_hw,
            Keys.intrinsics,
            Keys.boxes2d,
            Keys.boxes2d_track_ids,
            Keys.boxes3d,
            Keys.segmentation_masks,
            Keys.depth_maps,
            Keys.points3d,
        ],
        views_to_load=["front", "left_90", "center"],
        backend=HDF5Backend(),
    )

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.dataset), 1)

    def test_len_zip(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.dataset_zip), 1)

    def test_len_multiview(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.dataset_multiview), 1)

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        for view in ("front", "left_90"):
            self.assertEqual(
                tuple(self.dataset_multiview[0][view].keys()),
                (
                    Keys.images,
                    Keys.input_hw,
                    Keys.intrinsics,
                    Keys.boxes2d,
                    Keys.boxes2d_track_ids,
                    Keys.boxes3d,
                    Keys.segmentation_masks,
                    Keys.depth_maps,
                ),
            )
            self.assertEqual(
                self.dataset_multiview[0][view][Keys.images].shape,
                (1, 3, 800, 1280),
            )
            self.assertEqual(
                self.dataset_multiview[0][view][Keys.segmentation_masks].shape,
                (1, 800, 1280),
            )
            self.assertEqual(
                self.dataset_multiview[0][view][Keys.depth_maps].shape,
                (1, 800, 1280),
            )

        for view in ("center",):
            self.assertEqual(
                tuple(self.dataset_multiview[0][view].keys()),
                (
                    Keys.boxes3d,
                    Keys.points3d,
                ),
            )
            self.assertEqual(
                self.dataset_multiview[0]["center"][Keys.points3d].shape,
                (51111, 4),
            )
