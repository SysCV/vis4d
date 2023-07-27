"""NuScenes monocular dataset testing class."""
import unittest

from tests.util import get_test_data
from vis4d.data.datasets.nuscenes_mono import NuScenesMono


class NuScenesMonoTest(unittest.TestCase):
    """Test NuScenes Monocular dataloading."""

    data_root = get_test_data("nuscenes_test", absolute_path=False)

    nusc = NuScenesMono(
        data_root=data_root,
        version="v1.0-mini",
        split="mini_train",
        cache_as_binary=True,
        cached_file_path=f"{data_root}/mono_mini_train.pkl",
    )

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.nusc), 1938)

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        assert tuple(self.nusc[1].keys()) == (
            "token",
            "sequence_names",
            "frame_ids",
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
        )
