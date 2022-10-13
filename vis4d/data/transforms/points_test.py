"""point sampling augmentation testing class."""
import copy
import unittest

import pytest
import torch

from vis4d.common import COMMON_KEYS

from .points import move_pts_to_last_channel, rotate_around_axis


class TestPoints(unittest.TestCase):
    """Tests sampling in a block based fashion."""

    @pytest.fixture(autouse=True)
    def initdata(self):
        """Loads dummy data."""
        self.data = {
            COMMON_KEYS.points3d: torch.rand(200, 3),
            COMMON_KEYS.colors3d: torch.rand(200, 3),
            COMMON_KEYS.semantics3d: torch.randint(10, (200, 1)),
        }
        self.original_data = copy.deepcopy(self.data)

    def test_move_pts_to_last_channel(self) -> None:
        """Tests the functional."""
        tf = move_pts_to_last_channel(
            in_keys=[COMMON_KEYS.points3d], out_keys=[COMMON_KEYS.points3d]
        )

        # Check that points are now at last channel
        self.assertEqual(tf(self.data)[COMMON_KEYS.points3d].shape[-1], 200)

    def test_move_pts_to_last_channel_w_multi_keys(self) -> None:
        """Tests the move_pts_to_last_channel functional with multiple inputs."""
        # Check mutli key case
        tf = move_pts_to_last_channel(
            in_keys=[
                COMMON_KEYS.points3d,
                COMMON_KEYS.colors3d,
                COMMON_KEYS.semantics3d,
            ],
            out_keys=[
                COMMON_KEYS.points3d,
                COMMON_KEYS.colors3d,
                COMMON_KEYS.semantics3d,
            ],
        )

        # Check that num_points are now at last channel
        out = tf(self.data)
        for data in out.values():
            self.assertEqual(data.shape[-1], 200)

    def test_rotation_not_rotate_points(self) -> None:
        """Tests rotation of pointcloud."""
        # No rotation
        tf = rotate_around_axis(angle_min=0, angle_max=0)
        out = tf(self.data)
        self.assertTrue(
            (
                out[COMMON_KEYS.points3d]
                == self.original_data[COMMON_KEYS.points3d]
            ).all()
        )
        self.assertEqual(1, 2)

    def test_rotate_points_180_deg(self) -> None:
        """Tests rotation of pointcloud."""
        # 180 degree rotation
        tf = rotate_around_axis(angle_min=torch.pi, angle_max=torch.pi, axis=2)
        out = tf(self.data)

        in_points = self.original_data[COMMON_KEYS.points3d]
        out_points = out[COMMON_KEYS.points3d]
        # Make sure signs are correct
        self.assertTrue(
            torch.isclose(in_points[:, :2], -out_points[:, :2]).all()
        )
        # Z component should not change
        self.assertTrue((in_points[:, -1] == out_points[:, -1]).all())
