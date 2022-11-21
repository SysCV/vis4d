"""Point transformation testing class."""
import copy
import unittest

import pytest
import torch

from vis4d.data.const import CommonKeys

from .points import move_pts_to_last_channel, rotate_around_axis


# TODO, more tests required here
class TestPoints(unittest.TestCase):
    """Tests sampling in a block based fashion."""

    @pytest.fixture(autouse=True)
    def initdata(self):
        """Loads dummy data."""
        self.data = {
            CommonKeys.points3d: torch.rand(200, 3),
            CommonKeys.colors3d: torch.rand(200, 3),
            CommonKeys.semantics3d: torch.randint(10, (200, 1)),
        }
        self.original_data = copy.deepcopy(self.data)

    def test_move_pts_to_last_channel(self) -> None:
        """Tests the functional."""
        # pylint: disable=unexpected-keyword-arg
        tf = move_pts_to_last_channel(
            in_keys=(CommonKeys.points3d,), out_keys=(CommonKeys.points3d,)
        )

        # Check that points are now at last channel
        self.assertEqual(tf(self.data)[CommonKeys.points3d].shape[-1], 200)

    def test_move_pts_to_last_channel_w_multi_keys(self) -> None:
        """Tests the move_pts_to_last_channel functional with multiple inputs."""
        # Check mutli key case
        # pylint: disable=unexpected-keyword-arg
        tf = move_pts_to_last_channel(
            in_keys=(
                CommonKeys.points3d,
                CommonKeys.colors3d,
                CommonKeys.semantics3d,
            ),
            out_keys=(
                CommonKeys.points3d,
                CommonKeys.colors3d,
                CommonKeys.semantics3d,
            ),
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
                out[CommonKeys.points3d]
                == self.original_data[CommonKeys.points3d]
            ).all()
        )

    def test_rotate_points_180_deg(self) -> None:
        """Tests rotation of pointcloud."""
        # 180 degree rotation
        tf = rotate_around_axis(angle_min=torch.pi, angle_max=torch.pi, axis=2)
        out = tf(self.data)

        in_points = self.original_data[CommonKeys.points3d]
        out_points = out[CommonKeys.points3d]
        # Make sure signs are correct
        self.assertTrue(
            torch.isclose(in_points[:, :2], -out_points[:, :2]).all()
        )
        # Z component should not change
        self.assertTrue((in_points[:, -1] == out_points[:, -1]).all())
