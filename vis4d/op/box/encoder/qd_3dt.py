"""3D bounding box coder."""
from __future__ import annotations

import torch
from torch import Tensor

from ...geometry.projection import unproject_points
from ...geometry.rotation import alpha2yaw, rotation_output_to_alpha
from .base import BoxEncoder3D


class QD3DTBox3DEncoder(BoxEncoder3D):
    """3D bounding box coder based on qd_3dt."""

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

    def encode(
        self,
        boxes_3d: Tensor,
        targets: Tensor,
        intrinsics: Tensor,
    ) -> Tensor:
        """Encode deltas between boxes and targets given intrinsics."""
        # result = []
        # for boxes_, targets_, intrinsics_ in zip(boxes, targets, intrinsics):
        #     # delta center 2d
        #     projected_3d_center = project_points(targets_.center, intrinsics_) # pylint: disable=line-too-long
        #     delta_center = (
        #         projected_3d_center - boxes_.center
        #     ) / self.center_scale

        #     # depth
        #     depth = torch.where(
        #         targets_.center[:, -1] > 0,
        #         torch.log(targets_.center[:, -1]) * self.depth_log_scale,
        #         -targets_.center[:, -1].new_ones(1),
        #     )
        #     depth = depth.unsqueeze(-1)

        #     # dimensions
        #     dims = torch.where(
        #         targets_.dimensions > 0,
        #         torch.log(targets_.dimensions) * self.dim_log_scale,
        #         targets_.dimensions.new_ones(1) * 100.0,
        #     )

        #     # rotation
        #     num_bins, bin_overlap = (
        #         self.num_rotation_bins,
        #         self.bin_overlap,
        #     )
        #     alpha = yaw2alpha(targets_.rot_y, targets_.center)
        #     bin_cls = torch.zeros(
        #         (alpha.shape[0], num_bins), device=alpha.device
        #     )
        #     bin_res = torch.zeros(
        #         (alpha.shape[0], num_bins), device=alpha.device
        #     )
        #     bin_centers = torch.arange(
        #         -np.pi, np.pi, 2 * np.pi / num_bins, device=alpha.device
        #     )
        #     bin_centers += np.pi / num_bins
        #     for i in range(alpha.shape[0]):
        #         overlap_value = np.pi * 2 / num_bins * bin_overlap
        #         alpha_hi = normalize_angle(alpha[i] + overlap_value)
        #         alpha_lo = normalize_angle(alpha[i] - overlap_value)
        #         for bin_idx in range(self.num_rotation_bins):
        #             bin_min = bin_centers[bin_idx] - np.pi / num_bins
        #             bin_max = bin_centers[bin_idx] + np.pi / num_bins
        #             if (
        #                 bin_min <= alpha_lo <= bin_max
        #                 or bin_min <= alpha_hi <= bin_max
        #             ):
        #                 bin_cls[i, bin_idx] = 1
        #                 bin_res[i, bin_idx] = alpha[i] - bin_centers[bin_idx]

        #     result.append(
        #         torch.cat([delta_center, depth, dims, bin_cls, bin_res], -1)
        #     )
        # return result
        raise NotImplementedError

    def decode(
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
