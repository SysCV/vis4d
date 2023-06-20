# pylint: disable=no-member,unexpected-keyword-arg,use-dict-literal
"""Point transformation testing class."""
import copy
import unittest

import numpy as np

from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms import compose
from vis4d.data.transforms.points import (
    ApplySE3Transform,
    ApplySO3Transform,
    TransposeChannels,
)


class TestPoints(unittest.TestCase):
    """Tests sampling in a block based fashion."""

    data = {K.points3d: np.random.rand(200, 3)}

    def test_move_pts_to_last_channel(self) -> None:
        """Tests the functional."""
        # pylint: disable=unexpected-keyword-arg
        transform = TransposeChannels(channels=(-1, -2))
        out = transform.apply_to_data([copy.deepcopy(self.data)])[0]
        self.assertEqual(out[K.points3d].shape, (3, 200))

    def test_no_se3_tf(self) -> None:
        """Tests rotation of pointcloud."""
        tf = ApplySE3Transform((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))

        out = tf.apply_to_data([copy.deepcopy(self.data)])[0]
        self.assertTrue(np.all(out[K.points3d] == self.data[K.points3d]))

        # Make sure also works if channels are not last
        swap_ch = TransposeChannels(channels=(-1, -2))
        transform = compose([swap_ch, tf])
        out = transform([copy.deepcopy(self.data)])[0]
        self.assertTrue(np.all(out[K.points3d] == self.data[K.points3d]))

    def test_rotate_points_180_deg(self) -> None:
        """Tests rotation of pointcloud of 180 deg. around z axis."""
        # 180 degree rotation
        tf = ApplySO3Transform(
            (0, 0, 0), (0, 0, 0), (0, 0, np.pi), (0, 0, np.pi)
        )
        out = tf.apply_to_data([copy.deepcopy(self.data)])[0]

        in_points = self.data[K.points3d]
        out_points = out[K.points3d]
        # Make sure signs are correct
        self.assertTrue(
            np.all(np.isclose(in_points[:, :2], -out_points[:, :2]))
        )
        # Z component should not change
        self.assertTrue(np.all(in_points[:, -1] == out_points[:, -1]))
