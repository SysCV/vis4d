"""NuScenes dataset testing class."""
import unittest

from tests.util import get_test_data
from vis4d.data.datasets.nuscenes import NuScenes


class NuScenesTest(unittest.TestCase):
    """Test NuScenes dataloading."""

    data_root = get_test_data("nuscenes_test", absolute_path=False)

    nusc = NuScenes(
        data_root=data_root,
        version="v1.0-mini",
        split="mini_val",
        cache_as_binary=True,
        cached_file_path=f"{data_root}/mini_val.pkl",
    )

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.nusc), 81)

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        self.assertCountEqual(
            list(self.nusc[0].keys()),
            [
                "token",
                "frame_ids",
                "sequence_names",
                "can_bus",
                "LIDAR_TOP",
                "CAM_FRONT",
                "CAM_FRONT_LEFT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ],
        )

        self.assertCountEqual(
            list(self.nusc[1]["CAM_FRONT"].keys()),
            [
                "timestamp",
                "images",
                "input_hw",
                "sample_names",
                "intrinsics",
                "boxes3d",
                "boxes3d_classes",
                "boxes3d_track_ids",
                "boxes3d_velocities",
                "attributes",
                "extrinsics",
                "axis_mode",
                "boxes2d",
                "boxes2d_classes",
                "boxes2d_track_ids",
            ],
        )
