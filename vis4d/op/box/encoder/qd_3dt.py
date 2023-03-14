"""3D bounding box coder."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ...geometry.projection import unproject_points
from ...geometry.rotation import alpha2yaw, rotation_output_to_alpha


# TODO: Add qd-3dt 3D box encoder for training
class QD3DTBox3DDecoder(nn.Module):
    """3D bounding box decoder based on qd_3dt."""

    def __init__(
        self,
        center_scale: float = 10.0,
        depth_log_scale: float = 2.0,
        dim_log_scale: float = 2.0,
        num_rotation_bins: int = 2,
        bin_overlap: float = 1 / 6,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.center_scale = center_scale
        self.depth_log_scale = depth_log_scale
        self.dim_log_scale = dim_log_scale
        self.num_rotation_bins = num_rotation_bins
        self.bin_overlap = bin_overlap

    def forward(
        self,
        boxes_2d: Tensor,
        boxes_deltas: Tensor,
        intrinsics: Tensor,
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
