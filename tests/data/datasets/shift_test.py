"""S3DIS dataset testing class."""
import unittest

from tests.util import get_test_data
from vis4d.data.const import CommonKeys
from vis4d.data.datasets.shift import SHIFT


class SHIFTTest(unittest.TestCase):
    """Test S3DIS dataloading."""

    dataset = SHIFT(data_root=get_test_data("shift_test"), split="val")

    dataset_multiview = SHIFT(
        data_root=get_test_data("shift_test"),
        split="val",
        keys_to_load=[
            # --- from Scalabel ---
            CommonKeys.images,
            CommonKeys.input_hw,
            CommonKeys.intrinsics,
            CommonKeys.boxes2d,
            CommonKeys.boxes2d_track_ids,
            CommonKeys.boxes3d,
            # --- from bit masks ---
            CommonKeys.depth_maps,
            # --- from ply files ---
            CommonKeys.points3d,
        ],
        views_to_load=["front", "left_90", "center"],
    )

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.dataset), 1)

    def test_len_multiview(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.dataset_multiview), 1)

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        for view in ["front", "left_90"]:
            self.assertEqual(
                tuple(self.dataset_multiview[0][view].keys()),
                (
                    CommonKeys.images,
                    CommonKeys.input_hw,
                    CommonKeys.intrinsics,
                    CommonKeys.boxes2d,
                    CommonKeys.boxes2d_track_ids,
                    CommonKeys.boxes3d,
                    CommonKeys.depth_maps,
                ),
            )

        for view in ["center"]:
            self.assertEqual(
                tuple(self.dataset_multiview[0][view].keys()),
                (CommonKeys.points3d,),
            )
