# pylint: disable=no-member,unexpected-keyword-arg,use-dict-literal
"""Point transformation testing class."""
import copy
import unittest
from typing import Any

import numpy as np

from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms import compose
from vis4d.data.transforms.points import (
    ApplySE3Transform,
    ApplySO3Transform,
    ColorContrast,
    ColorDrop,
    ColorNormalize,
    GenContrastParams,
    GenScaleParams,
    PointJitter,
    PointScale,
    TransposeChannels,
    XYCenterZAlign,
)


def apply_to_data(
    data: dict[str, np.ndarray], transforms: list[Any]  # type: ignore
) -> dict[str, np.ndarray]:
    """Applies the transform to the data."""
    tf = compose(transforms)
    return tf([copy.deepcopy(data)])[0]


class TestPoints(unittest.TestCase):
    """Tests sampling in a block based fashion."""

    data = {
        K.points3d: np.random.rand(200, 3),
        K.colors3d: np.random.rand(200, 3),
    }

    def test_align(self) -> None:
        """Tests alignment."""
        out = apply_to_data(self.data, [XYCenterZAlign()])
        self.assertAlmostEqual(out[K.points3d][:, 2].min(), 0)
        self.assertAlmostEqual(out[K.points3d][:, 0].mean(), 0)
        self.assertAlmostEqual(out[K.points3d][:, 1].mean(), 0)

    def test_jitter(self) -> None:
        """Tests jitter."""
        out = apply_to_data(self.data, [PointJitter()])
        self.assertEqual(out[K.points3d].shape, (200, 3))

    def test_color_normalize(self) -> None:
        """Tests color normalization."""
        out = apply_to_data(
            self.data,
            [
                ColorNormalize(
                    color_mean=np.mean(self.data[K.colors3d], axis=0),
                    color_std=np.std(self.data[K.colors3d], axis=0),
                )
            ],
        )
        self.assertAlmostEqual(out[K.colors3d].mean(), 0)
        self.assertAlmostEqual(out[K.colors3d].std(), 1)

        out = apply_to_data(
            self.data,
            [ColorNormalize(None, None)],
        )
        self.assertTrue(np.allclose(out[K.colors3d], self.data[K.colors3d]))

        # Check normalization

        out = apply_to_data(
            {K.colors3d: self.data[K.colors3d] * 255},
            [ColorNormalize(None, None)],
        )
        self.assertTrue(np.allclose(out[K.colors3d], self.data[K.colors3d]))

    def test_color_contrast(self) -> None:
        """Tests color contrast."""
        p_gen = GenContrastParams(proba=1, blend_factor=0)
        out = apply_to_data(self.data, [p_gen, ColorContrast()])
        self.assertTrue(np.all(out[K.colors3d] == self.data[K.colors3d]))

        p_gen = GenContrastParams(proba=1, blend_factor=1)
        out = apply_to_data(self.data, [p_gen, ColorContrast()])
        # Full contrast, max colors should be from 0 to 1
        self.assertAlmostEqual(out[K.colors3d].max(), 1)
        self.assertAlmostEqual(out[K.colors3d].min(), 0)

    def test_color_drop(self) -> None:
        """Tests dropping of color channels."""
        # 0 color drop probability
        out = apply_to_data(self.data, [ColorDrop(0)])
        self.assertTrue(np.all(out[K.colors3d] == self.data[K.colors3d]))

        out = apply_to_data(self.data, [ColorDrop(1)])
        self.assertTrue(np.all(out[K.colors3d] == 0 * self.data[K.colors3d]))

        self.assertEqual(out[K.colors3d].shape, (200, 3))

    def test_point_scale(self) -> None:
        """Tests scaling of pointcloud."""
        p_gen = GenScaleParams(
            scale=(1, 1),
            scale_anisotropic=True,
            scale_xyz=(True, True, True),
            mirror=(0, 0, 0),
        )
        scale_tf = PointScale()
        out = apply_to_data(self.data, [p_gen, scale_tf])
        self.assertTrue(np.all(out[K.points3d] == self.data[K.points3d]))

        # Test scale * 2
        p_gen = GenScaleParams(
            scale=(2, 2),
            scale_anisotropic=True,
            scale_xyz=(True, True, True),
            mirror=(0, 0, 0),
        )
        out = apply_to_data(self.data, [p_gen, scale_tf])
        self.assertTrue(np.all(out[K.points3d] == 2 * self.data[K.points3d]))

        # Test mirror
        p_gen = GenScaleParams(
            scale=(1, 1),
            scale_anisotropic=True,
            scale_xyz=(True, True, True),
            mirror=(1, 0, 0),
        )
        out = apply_to_data(self.data, [p_gen, scale_tf])

        self.assertTrue(
            np.all(out[K.points3d][:, 0] == -self.data[K.points3d][:, 0])
        )
        self.assertTrue(
            np.all(out[K.points3d][:, 1:] == self.data[K.points3d][:, 1:])
        )

        # check not scaling z
        p_gen = GenScaleParams(
            scale=(2, 2),
            scale_anisotropic=False,
            scale_xyz=(True, True, False),
            mirror=(0, 0, 0),
        )
        out = apply_to_data(self.data, [p_gen, scale_tf])
        self.assertTrue(
            np.all(
                out[K.points3d][:, :-1] == 2 * self.data[K.points3d][:, :-1]
            )
        )
        self.assertTrue(
            np.all(out[K.points3d][:, -1] == self.data[K.points3d][:, -1])
        )

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
