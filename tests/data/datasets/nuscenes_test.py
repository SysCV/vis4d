"""NuScenes dataset testing class."""
import unittest

from tests.util import get_test_data
from vis4d.data.datasets.nuscenes import NuScenes


class NuScenesTest(unittest.TestCase):
    """Test NuScenes dataloading."""

    nusc = NuScenes(
        data_root=get_test_data("nuscenes_test"),
        version="v1.0-mini",
        split="mini_val",
        metadata=["use_camera"],
    )

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.nusc), 81)

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        assert tuple(self.nusc[0].keys()) == (
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        )

        assert tuple(self.nusc[1]["CAM_FRONT"].keys()) == (
            "token",
            "images",
            "original_hw",
            "input_hw",
            "frame_ids",
            "intrinsics",
            "extrinsics",
            "timestamp",
            "axis_mode",
            "boxes2d",
            "boxes2d_classes",
            "boxes2d_track_ids",
            "boxes3d",
            "boxes3d_classes",
            "boxes3d_track_ids",
        )
