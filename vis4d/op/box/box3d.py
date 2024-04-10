"""Utility functions for 3D bounding boxes."""

from __future__ import annotations

import torch
from torch import Tensor

from vis4d.data.const import AxisMode
from vis4d.op.geometry.projection import project_points
from vis4d.op.geometry.rotation import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
    rotate_orientation,
    rotation_matrix_yaw,
)
from vis4d.op.geometry.transform import get_transform_matrix, transform_points


def boxes3d_to_corners(boxes3d: Tensor, axis_mode: AxisMode) -> Tensor:
    """Convert a Tensor of 3D boxes to its respective corner points.

    Args:
        boxes3d (Tensor): Box parameters. Tensor of shape [N, 10].
        axis_mode (AxisMode): Coordinate system convention.

    Returns:
        Tensor: [N, 8, 3] 3D bounding box corner coordinates, in this order:

               (back)
        (6) +---------+. (7)
            | ` .     |  ` .
            | (4) +---+-----+ (5)
            |     |   |     |
        (2) +-----+---+. (3)|
            ` .   |     ` . |
            (0) ` +---------+ (1)
                     (front)
    """
    w, l, h = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    rotation_matrix = quaternion_to_matrix(boxes3d[:, 6:])

    if axis_mode == AxisMode.OPENCV:
        x_corners = torch.stack(
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            dim=-1,
        )
        y_corners = torch.stack(
            [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2],
            dim=-1,
        )
        z_corners = torch.stack(
            [-w / 2, w / 2, -w / 2, w / 2, -w / 2, w / 2, -w / 2, w / 2],
            dim=-1,
        )
    else:
        x_corners = torch.stack(
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            dim=-1,
        )
        y_corners = torch.stack(
            [-w / 2, w / 2, -w / 2, w / 2, -w / 2, w / 2, -w / 2, w / 2],
            dim=-1,
        )
        z_corners = torch.stack(
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
            dim=-1,
        )

    corners = torch.stack([x_corners, y_corners, z_corners], dim=-1)
    corners = transform_points(
        corners, get_transform_matrix(rotation_matrix, boxes3d[:, :3])
    )
    return corners


def boxes3d_in_image(
    box_corners: Tensor, cam_intrinsics: Tensor, image_hw: tuple[int, int]
) -> Tensor:
    """Check if a 3D bounding box is (partially) in an image.

    Args:
        box_corners (Tensor): [N, 8, 3] Tensor of 3D boxes corners. In OpenCV
            coordinate frame.
        cam_intrinsics (Tensor): [3, 3] Camera matrix.
        image_hw (tuple[int, int]): image height / width.

    Returns:
        Tensor: [N,] boolean values.
    """
    points = project_points(box_corners.view(-1, 3), cam_intrinsics).view(
        -1, 8, 2
    )
    mask = (points[..., 0] >= 0) * (points[..., 0] < image_hw[1]) * (
        points[..., 1] >= 0
    ) * (points[..., 1] < image_hw[0]) * box_corners[..., 2] > 0.0
    mask = mask.any(dim=-1)
    return mask


def transform_boxes3d(
    boxes3d: Tensor,
    transform_matrix: Tensor,
    source_axis_mode: AxisMode,
    target_axis_mode: AxisMode,
    only_yaw: bool = True,
) -> Tensor:
    """Transform 3D boxes using given transform matrix.

    Args:
        boxes3d (Tensor): [N, 10] Tensor of 3D boxes.
        transform_matrix (Tensor): [4, 4] Transform matrix.
        source_axis_mode (AxisMode): Source coordinate system convention of the
            boxes.
        target_axis_mode (AxisMode): Target coordinate system convention of the
            boxes.
        only_yaw (bool): Whether to only care about yaw rotation.
    """
    boxes3d_transformed = boxes3d.new_zeros(boxes3d.shape)
    boxes3d_transformed[:, :3] = transform_points(
        boxes3d[:, :3], transform_matrix
    )
    boxes3d_transformed[:, 3:6] = boxes3d[:, 3:6]

    if only_yaw:
        orientation = rotation_matrix_yaw(
            quaternion_to_matrix(boxes3d[:, 6:]), source_axis_mode
        )

        orientation = rotate_orientation(
            orientation, transform_matrix, axis_mode=target_axis_mode
        )

        boxes3d_transformed[:, 6:] = matrix_to_quaternion(
            euler_angles_to_matrix(orientation)
        )
    else:
        rot_quat = matrix_to_quaternion(transform_matrix[:3, :3])
        boxes3d_transformed[:, 6:] = quaternion_multiply(
            rot_quat, boxes3d[:, 6:]
        )

    return boxes3d_transformed
