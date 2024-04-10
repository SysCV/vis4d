"""NuScenes trajectory dataset tests."""

import unittest

from tests.util import get_test_data
from vis4d.data.datasets.nuscenes_trajectory import NuScenesTrajectory


class NuScenesTrajectoryTest(unittest.TestCase):
    """Testclass for NuScenes trajectory dataset."""

    data_root = get_test_data("nuscenes_test", absolute_path=False)

    nusc = NuScenesTrajectory(
        detector="test",
        pure_detection=f"{data_root}/cc_3dt_pure_det.json",
        data_root=data_root,
        version="v1.0-mini",
        split="mini_val",
    )

    def test_getitem(self):
        """Test dataset getitem."""
        data = self.nusc[0]

        assert "gt_traj" in data and "pred_traj" in data

        self.assertEqual(data["gt_traj"].shape, (10, 8))
        self.assertEqual(data["pred_traj"].shape, (10, 8))
