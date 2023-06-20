"""Utility functions for 3D bounding boxes."""
from __future__ import annotations

import torch
from torch import Tensor

from vis4d.data.const import AxisMode
from vis4d.op.geometry.projection import project_points
from vis4d.op.geometry.rotation import quaternion_to_matrix
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

    if axis_mode == AxisMode.ROS:
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
    elif axis_mode == AxisMode.OPENCV:
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

    corners = torch.stack([x_corners, y_corners, z_corners], dim=-1)
    corners = transform_points(
        corners, get_transform_matrix(rotation_matrix, boxes3d[:, :3])
    )
    return corners


def boxes3d_in_image(
    boxes: torch.Tensor,
    cam_intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    """Check if a 3D bounding box is (partially) in an image.

    Args:
        boxes (torch.Tensor): [N, 10] Tensor of 3D boxes. In OpenCV coordinate
            frame.
        cam_intrinsics (torch.Tensor): [3, 3] Camera matrix.
        image_hw (tuple[int, int]): image height / width.

    Returns:
        torch.Tensor: [N,] boolean values.
    """
    box_corners = boxes3d_to_corners(boxes, AxisMode.OPENCV)
    points = project_points(box_corners.view(-1, 3), cam_intrinsics).view(
        -1, 8, 2
    )
    mask = (points[..., 0] >= 0) * (points[..., 0] < image_hw[1]) * (
        points[..., 1] >= 0
    ) * (points[..., 1] < image_hw[0]) * box_corners[..., 2] > 0.0
    mask = mask.any(dim=-1)
    return mask
