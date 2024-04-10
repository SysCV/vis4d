"""3D bounding box coder."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from vis4d.data.const import AxisMode
from vis4d.op.geometry.projection import project_points, unproject_points
from vis4d.op.geometry.rotation import (
    alpha2yaw,
    normalize_angle,
    quaternion_to_matrix,
    rotation_matrix_yaw,
    rotation_output_to_alpha,
    yaw2alpha,
)


class QD3DTBox3DEncoder:
    """3D bounding box encoder based on qd_3dt."""

    def __init__(
        self,
        center_scale: float = 10.0,
        depth_log_scale: float = 2.0,
        dim_log_scale: float = 2.0,
        num_rotation_bins: int = 2,
        bin_overlap: float = 1 / 6,
    ) -> None:
        """Init."""
        self.center_scale = center_scale
        self.depth_log_scale = depth_log_scale
        self.dim_log_scale = dim_log_scale
        self.num_rotation_bins = num_rotation_bins
        self.bin_overlap = bin_overlap

    def __call__(
        self, boxes: Tensor, boxes3d: Tensor, intrinsics: Tensor
    ) -> Tensor:
        """Encode deltas between 2D boxes and 3D boxes given intrinsics."""
        # delta center 2d
        projected_center_3d = project_points(boxes3d[:, :3], intrinsics)
        ctr_x = (boxes[:, 0] + boxes[:, 2]) / 2
        ctr_y = (boxes[:, 1] + boxes[:, 3]) / 2
        center_2d = torch.stack([ctr_x, ctr_y], -1)
        delta_center = (projected_center_3d - center_2d) / self.center_scale

        # depth
        depth = torch.where(
            boxes3d[:, 2] > 0,
            torch.log(boxes3d[:, 2]) * self.depth_log_scale,
            -boxes3d[:, 2].new_ones(1),
        )
        depth = depth.unsqueeze(-1)

        # dimensions
        dims = torch.where(
            boxes3d[:, 3:6] > 0,
            torch.log(boxes3d[:, 3:6]) * self.dim_log_scale,
            boxes3d[:, 3:6].new_ones(1) * 100.0,
        )

        # WLH -> HWL
        dims = dims[:, [2, 0, 1]]

        # rotation
        yaw = rotation_matrix_yaw(
            quaternion_to_matrix(boxes3d[:, 6:]), axis_mode=AxisMode.OPENCV
        )[:, 1]
        alpha = yaw2alpha(yaw, boxes3d[:, :3])
        bin_cls = torch.zeros(
            (alpha.shape[0], self.num_rotation_bins), device=alpha.device
        )
        bin_res = torch.zeros(
            (alpha.shape[0], self.num_rotation_bins), device=alpha.device
        )
        bin_centers = torch.arange(
            -np.pi,
            np.pi,
            2 * np.pi / self.num_rotation_bins,
            device=alpha.device,
        )
        bin_centers += np.pi / self.num_rotation_bins
        for i in range(alpha.shape[0]):
            overlap_value = (
                np.pi * 2 / self.num_rotation_bins * self.bin_overlap
            )
            alpha_hi = normalize_angle(alpha[i] + overlap_value)
            alpha_lo = normalize_angle(alpha[i] - overlap_value)
            for bin_idx in range(self.num_rotation_bins):
                bin_min = bin_centers[bin_idx] - np.pi / self.num_rotation_bins
                bin_max = bin_centers[bin_idx] + np.pi / self.num_rotation_bins
                if (
                    bin_min <= alpha_lo <= bin_max
                    or bin_min <= alpha_hi <= bin_max
                ):
                    bin_cls[i, bin_idx] = 1
                    bin_res[i, bin_idx] = alpha[i] - bin_centers[bin_idx]

        return torch.cat([delta_center, depth, dims, bin_cls, bin_res], -1)


class QD3DTBox3DDecoder:
    """3D bounding box decoder based on qd_3dt."""

    def __init__(
        self,
        center_scale: float = 10.0,
        depth_log_scale: float = 2.0,
        dim_log_scale: float = 2.0,
        num_rotation_bins: int = 2,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.center_scale = center_scale
        self.depth_log_scale = depth_log_scale
        self.dim_log_scale = dim_log_scale
        self.num_rotation_bins = num_rotation_bins

    def __call__(
        self, boxes_2d: Tensor, boxes_deltas: Tensor, intrinsics: Tensor
    ) -> Tensor:
        """Decode the predicted boxes_deltas according to given 2D boxes."""
        # center
        delta_center = boxes_deltas[:, 0:2] * self.center_scale
        ctr_x = (boxes_2d[:, 0] + boxes_2d[:, 2]) / 2
        ctr_y = (boxes_2d[:, 1] + boxes_2d[:, 3]) / 2
        boxes_2d_center = torch.stack([ctr_x, ctr_y], -1)
        center_2d = boxes_2d_center + delta_center
        depth = torch.exp(boxes_deltas[:, 2:3] / self.depth_log_scale)
        center_3d = unproject_points(center_2d, depth, intrinsics)

        # dimensions
        dimensions = torch.exp(boxes_deltas[:, 3:6] / self.dim_log_scale)

        # rot_y
        alpha = rotation_output_to_alpha(
            boxes_deltas[:, 6:-1], self.num_rotation_bins
        )
        rot_y = alpha2yaw(alpha, center_3d)
        orientation = torch.stack(
            [torch.zeros_like(rot_y), rot_y, torch.zeros_like(rot_y)], -1
        )

        velocities = torch.zeros(
            (boxes_deltas.shape[0], 3), device=boxes_deltas.device
        )

        return torch.cat(
            [
                center_3d,
                dimensions,
                orientation,
                velocities,
            ],
            1,
        )
