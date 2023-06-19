# pylint: disable=no-member
"""Test Flip transform."""
import copy
import unittest

import numpy as np
import torch

from vis4d.data.const import AxisMode
from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.flip import (
    FlipBoxes2D,
    FlipBoxes3D,
    FlipImages,
    FlipIntrinsics,
    FlipPoints3D,
    FlipSegMasks,
)
from vis4d.op.geometry.rotation import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)


class TestFlip(unittest.TestCase):
    """Test Flip transform."""

    data = {
        K.images: np.random.rand(1, 16, 16, 3),
        K.boxes2d: np.random.rand(3, 4),
        K.seg_masks: np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
    }

    def test_flip_image(self):
        """Test the FlipImage transform."""
        transform = FlipImages(direction="horizontal")
        images_tr = transform.apply_to_data([copy.deepcopy(self.data)])[0][
            K.images
        ]
        assert images_tr.shape == (1, 16, 16, 3)
        assert (images_tr == self.data[K.images][:, :, ::-1, :]).all()

        transform = FlipImages(direction="vertical")
        images_tr = transform.apply_to_data([copy.deepcopy(self.data)])[0][
            K.images
        ]
        assert images_tr.shape == (1, 16, 16, 3)
        assert (images_tr == self.data[K.images][:, ::-1, :, :]).all()

    def test_flip_boxes2d(self):
        """Test the FlipBoxes2D transform."""
        transform = FlipBoxes2D(direction="horizontal")
        boxes_tr = transform.apply_to_data([copy.deepcopy(self.data)])[0][
            K.boxes2d
        ]
        assert boxes_tr.shape == (3, 4)
        assert (boxes_tr[:, 1::2] == self.data[K.boxes2d][:, 1::2]).all()
        assert (
            boxes_tr[:, 0::2] == np.flip(16 - self.data[K.boxes2d][:, 0::2], 1)
        ).all()

        transform = FlipBoxes2D(direction="vertical")
        boxes_tr = transform.apply_to_data([copy.deepcopy(self.data)])[0][
            K.boxes2d
        ]
        assert boxes_tr.shape == (3, 4)
        assert (boxes_tr[:, 0::2] == self.data[K.boxes2d][:, 0::2]).all()
        assert (
            boxes_tr[:, 1::2] == np.flip(16 - self.data[K.boxes2d][:, 1::2], 1)
        ).all()

    def test_flip_seg_masks(self):
        """Test the FlipSegMasks transform."""
        transform = FlipSegMasks(direction="horizontal")
        flipped_seg_mask = transform.apply_to_data([copy.deepcopy(self.data)])[
            0
        ][K.seg_masks]
        assert np.all(
            flipped_seg_mask == np.array([[2, 1, 0], [5, 4, 3], [8, 7, 6]])
        )

        transform = FlipSegMasks(direction="vertical")
        flipped_seg_mask = transform.apply_to_data([copy.deepcopy(self.data)])[
            0
        ][K.seg_masks]
        assert np.all(
            flipped_seg_mask == np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
        )

    def test_flip_3d(self):
        """Test the 3D related transforms."""
        quat = (
            matrix_to_quaternion(
                euler_angles_to_matrix(torch.tensor([0, np.pi / 4, 0]))
            )
            .numpy()
            .tolist()
        )
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        boxes = np.array([[1, 0, 0, 1, 1, 1, *quat]], dtype=np.float32)
        intrinsics = np.array(
            [[1, 0, 16], [0, 1, 16], [0, 0, 1]], dtype=np.float32
        )
        images = np.random.rand(1, 30, 30, 3).astype(np.float32)
        data = {
            K.points3d: points,
            K.boxes3d: boxes,
            K.intrinsics: intrinsics,
            K.images: images,
            K.axis_mode: AxisMode.OPENCV,
        }
        transform = compose(
            [
                FlipPoints3D(direction="horizontal"),
                FlipIntrinsics(direction="horizontal"),
                FlipBoxes3D(direction="horizontal"),
            ]
        )
        data_tr = transform([copy.deepcopy(data)])[0]
        assert (data_tr[K.points3d][:, 0] == -points[:, 0]).all()
        target_quat = matrix_to_quaternion(
            euler_angles_to_matrix(torch.tensor([[0, np.pi - np.pi / 4, 0]]))
        )
        assert torch.isclose(
            torch.from_numpy(data_tr[K.boxes3d][:, 6:]), target_quat, atol=1e-4
        ).all()
        assert (data_tr[K.boxes3d][:, 0] == -boxes[:, 0]).all()
        assert (data_tr[K.intrinsics][0, 2] == 14).all()
        assert (data_tr[K.intrinsics][1, 2] == 16).all()

        transform = compose(
            [
                FlipPoints3D(direction="vertical"),
                FlipIntrinsics(direction="vertical"),
                FlipBoxes3D(direction="vertical"),
            ]
        )
        data_tr = transform([copy.deepcopy(data)])[0]
        assert (data_tr[K.points3d][:, 1] == -points[:, 1]).all()
        target_quat = matrix_to_quaternion(
            euler_angles_to_matrix(torch.tensor([[0, np.pi / 4, np.pi]]))
        )
        assert torch.isclose(
            torch.from_numpy(data_tr[K.boxes3d][:, 6:]), target_quat, atol=1e-4
        ).all()
        assert (data_tr[K.boxes3d][:, 1] == -boxes[:, 1]).all()
        assert (data_tr[K.intrinsics][0, 2] == 16).all()
        assert (data_tr[K.intrinsics][1, 2] == 14).all()

    def test_wrong_direction(self):
        """Test the wrong direction."""
        with self.assertRaises(ValueError):
            FlipImages(direction="wrong")
        with self.assertRaises(ValueError):
            FlipBoxes2D(direction="wrong")
        with self.assertRaises(ValueError):
            FlipSegMasks(direction="wrong")
        with self.assertRaises(ValueError):
            FlipBoxes3D(direction="wrong")(
                [np.random.rand(16, 10)], [AxisMode.OPENCV]
            )
        with self.assertRaises(ValueError):
            FlipPoints3D(direction="wrong")(
                [np.random.rand(16, 3)], [AxisMode.OPENCV]
            )
        with self.assertRaises(ValueError):
            FlipIntrinsics(direction="wrong")
