# pylint: disable=no-member,unexpected-keyword-arg,use-dict-literal
"""Point transformation testing class."""
import unittest

import numpy as np
import pytest
import torch

from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms.points import (
    ApplySE3Transform,
    GenRandSE3Transform,
    TransposeChannels,
)


class TestPoints(unittest.TestCase):
    """Tests sampling in a block based fashion."""

    data: dict[str, torch.Tensor] = {}
    original_data: dict[str, torch.Tensor] = {}

    @pytest.fixture(autouse=True)
    def initdata(self):
        """Loads dummy data."""
        self.data = {K.points3d: np.random.rand(200, 3)}
        self.original_data = self.data.copy()

    def test_move_pts_to_last_channel(self) -> None:
        """Tests the functional."""
        # pylint: disable=unexpected-keyword-arg
        transform = TransposeChannels(channels=(-1, -2))
        out = transform.apply_to_data(self.data.copy())
        self.assertEqual(out[K.points3d].shape, (3, 200))

    def test_no_se3_tf(self) -> None:
        """Tests rotation of pointcloud."""
        transform = GenRandSE3Transform(
            (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)
        )
        tf = ApplySE3Transform()

        out = tf.apply_to_data(transform.apply_to_data(self.data.copy()))
        self.assertTrue(
            np.all(out[K.points3d] == self.original_data[K.points3d])
        )
        # Make sure also works if channels are not last
        swap_ch = TransposeChannels(channels=(-1, -2))
        out = tf.apply_to_data(
            swap_ch.apply_to_data(transform.apply_to_data(self.data.copy()))
        )
        self.assertTrue(
            np.all(out[K.points3d] == self.original_data[K.points3d])
        )

    def test_rotate_points_180_deg(self) -> None:
        """Tests rotation of pointcloud of 180 deg. around z axis."""
        # 180 degree rotation
        transform = GenRandSE3Transform(
            (0, 0, 0), (0, 0, 0), (0, 0, np.pi), (0, 0, np.pi)
        )
        tf = ApplySE3Transform()
        out = tf.apply_to_data(transform.apply_to_data(self.data.copy()))

        in_points = self.data[K.points3d]
        out_points = out[K.points3d]
        # Make sure signs are correct
        self.assertTrue(
            np.all(np.isclose(in_points[:, :2], -out_points[:, :2]))
        )
        # Z component should not change
        self.assertTrue(np.all(in_points[:, -1] == out_points[:, -1]))
