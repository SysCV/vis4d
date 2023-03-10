"""S3DIS dataset testing class."""
import unittest

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.shift import SHIFT
from vis4d.data.io import HDF5Backend, ZipBackend


class SHIFTTest(unittest.TestCase):
    """Test SHIFT dataloading."""

    dataset = SHIFT(data_root=get_test_data("shift_test"), split="val")

    dataset_zip = SHIFT(
        data_root=get_test_data("shift_test"),
        split="val",
        keys_to_load=[
            K.images,
            K.input_hw,
            K.intrinsics,
            K.boxes2d,
            K.segmentation_masks,
            K.depth_maps,
            K.optical_flows,
        ],
        backend=ZipBackend(),
    )

    dataset_multiview = SHIFT(
        data_root=get_test_data("shift_test"),
        split="val",
        keys_to_load=[
            K.images,
            K.input_hw,
            K.intrinsics,
            K.boxes2d,
            K.boxes2d_classes,
            K.boxes2d_track_ids,
            K.boxes3d,
            K.instance_masks,
            K.segmentation_masks,
            K.depth_maps,
            K.optical_flows,
            K.points3d,
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
                    K.images,
                    K.input_hw,
                    K.intrinsics,
                    K.boxes2d,
                    K.boxes2d_classes,
                    K.boxes2d_track_ids,
                    K.boxes3d,
                    K.instance_masks,
                    K.segmentation_masks,
                    K.depth_maps,
                    K.optical_flows,
                ),
            )
            self.assertEqual(
                self.dataset_multiview[0][view][K.images].shape,
                (1, 3, 800, 1280),
            )
            self.assertEqual(
                self.dataset_multiview[0][view][K.instance_masks].shape,
                (2, 800, 1280) if view == "front" else (0, 0, 0),
            )
            self.assertEqual(
                self.dataset_multiview[0][view][K.segmentation_masks].shape,
                (1, 800, 1280),
            )
            self.assertEqual(
                self.dataset_multiview[0][view][K.depth_maps].shape,
                (1, 800, 1280),
            )
            self.assertEqual(
                self.dataset_multiview[0][view][K.optical_flows].shape,
                (1, 2, 800, 1280),
            )

        for view in ("center",):
            self.assertEqual(
                tuple(self.dataset_multiview[0][view].keys()),
                (
                    K.boxes3d,
                    K.points3d,
                ),
            )
            self.assertEqual(
                self.dataset_multiview[0]["center"][K.points3d].shape,
                (51111, 4),
            )
