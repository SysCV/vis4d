"""Test Flip transform."""
import copy

import numpy as np
import torch

from vis4d.data.const import AxisMode
from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.flip import (
    FlipBoxes2D,
    FlipBoxes3D,
    FlipImage,
    FlipIntrinsics,
    FlipPoints3D,
)
from vis4d.op.geometry.rotation import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)


def test_flip_image():
    """Test the FlipImage transform."""
    images = np.random.rand(1, 16, 16, 3)

    transform = FlipImage(direction="horizontal")
    images_tr = transform(copy.deepcopy(images))
    assert images_tr.shape == (1, 16, 16, 3)
    assert (images_tr == images[:, :, ::-1, :]).all()

    transform = FlipImage(direction="vertical")
    images_tr = transform(copy.deepcopy(images))
    assert images_tr.shape == (1, 16, 16, 3)
    assert (images_tr == images[:, ::-1, :, :]).all()


def test_flip_boxes2d():
    """Test the FlipBoxes2D transform."""
    images = np.random.rand(1, 16, 16, 3)
    boxes = np.random.rand(3, 4)

    transform = FlipBoxes2D(direction="horizontal")
    boxes_tr = transform(copy.deepcopy(boxes), copy.deepcopy(images))
    assert boxes_tr.shape == (3, 4)
    assert (boxes_tr[:, 1::2] == boxes[:, 1::2]).all()
    assert (boxes_tr[:, 0::2] == np.flip(16 - boxes[:, 0::2], 1)).all()

    transform = FlipBoxes2D(direction="vertical")
    boxes_tr = transform(copy.deepcopy(boxes), copy.deepcopy(images))
    assert boxes_tr.shape == (3, 4)
    assert (boxes_tr[:, 0::2] == boxes[:, 0::2]).all()
    assert (boxes_tr[:, 1::2] == np.flip(16 - boxes[:, 1::2], 1)).all()


def test_flip_3d():
    """Test the 3D related transforms."""
    quat = (
        matrix_to_quaternion(
            euler_angles_to_matrix(torch.tensor([0, np.pi / 4, 0]))
        )
        .numpy()
        .tolist()
    )
    target_quat = matrix_to_quaternion(
        euler_angles_to_matrix(torch.tensor([[0, np.pi - np.pi / 4, 0]]))
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
    data_tr = transform(copy.deepcopy(data))
    assert (data_tr[K.points3d][:, 0] == -points[:, 0]).all()
    assert torch.isclose(
        torch.from_numpy(data_tr[K.boxes3d][:, 6:]), target_quat, atol=1e-4
    ).all()
    assert (data_tr[K.boxes3d][:, 0] == -boxes[:, 0]).all()
    assert (data_tr[K.intrinsics][0, 2] == 14).all()
    assert (data_tr[K.intrinsics][1, 2] == 16).all()
