"""3D bounding box coder."""
from typing import List

import numpy as np
import torch

from vis4d.struct import Boxes2D, Boxes3D, Intrinsics

from ...geometry.projection import project_points, unproject_points
from ...geometry.rotation import (
    alpha2yaw,
    normalize_angle,
    rotation_output_to_alpha,
    yaw2alpha,
)
from .base import BaseBoxCoder3D


class QD3DTBox3DCoder(BaseBoxCoder3D):
    """3D bounding box coder based on qd-3dt."""

    def __init__(
        self,
        center_scale: float = 10.0,
        depth_log_scale: float = 2.0,
        dim_log_scale: float = 2.0,
        num_rotation_bins: int = 2,
        bin_overlap: float = 1 / 6,
    ) -> None:
        """Init."""
        super().__init__()
        self.center_scale = center_scale
        self.depth_log_scale = depth_log_scale
        self.dim_log_scale = dim_log_scale
        self.num_rotation_bins = num_rotation_bins
        self.bin_overlap = bin_overlap

    def encode(
        self,
        boxes: List[Boxes2D],
        targets: List[Boxes3D],
        intrinsics: Intrinsics,
    ) -> List[torch.Tensor]:
        """Encode deltas between boxes and targets given intrinsics."""
        result = []
        for boxes_, targets_, intrinsics_ in zip(boxes, targets, intrinsics):
            # delta center 2d
            projected_3d_center = project_points(targets_.center, intrinsics_)
            delta_center = (
                projected_3d_center - boxes_.center
            ) / self.center_scale

            # depth
            depth = torch.where(
                targets_.center[:, -1] > 0,
                torch.log(targets_.center[:, -1]) * self.depth_log_scale,
                -targets_.center[:, -1].new_ones(1),
            )
            depth = depth.unsqueeze(-1)

            # dimensions
            dims = torch.where(
                targets_.dimensions > 0,
                torch.log(targets_.dimensions) * self.dim_log_scale,
                targets_.dimensions.new_ones(1),
            )

            # rotation
            num_bins, bin_overlap = (
                self.num_rotation_bins,
                self.bin_overlap,
            )
            alpha = yaw2alpha(targets_.rot_y, targets_.center)
            bin_cls = torch.zeros(
                (alpha.shape[0], num_bins), device=alpha.device
            )
            bin_res = torch.zeros(
                (alpha.shape[0], num_bins), device=alpha.device
            )
            bin_centers = torch.arange(
                -np.pi, np.pi, 2 * np.pi / num_bins, device=alpha.device
            )
            bin_centers += np.pi / num_bins
            for i in range(alpha.shape[0]):
                overlap_value = np.pi * 2 / num_bins * bin_overlap
                alpha_hi = normalize_angle(alpha[i] + overlap_value)
                alpha_lo = normalize_angle(alpha[i] - overlap_value)
                for bin_idx in range(self.num_rotation_bins):
                    bin_min = bin_centers[bin_idx] - np.pi / num_bins
                    bin_max = bin_centers[bin_idx] + np.pi / num_bins
                    if (
                        bin_min <= alpha_lo <= bin_max
                        or bin_min <= alpha_hi <= bin_max
                    ):
                        bin_cls[i, bin_idx] = 1
                        bin_res[i, bin_idx] = alpha[i] - bin_centers[bin_idx]

            result.append(
                torch.cat([delta_center, depth, dims, bin_cls, bin_res], -1)
            )

        return result

    def decode(
        self,
        boxes: List[Boxes2D],
        box_deltas: List[torch.Tensor],
        intrinsics: Intrinsics,
    ) -> List[Boxes3D]:
        """Decode the predicted box_deltas according to given base boxes."""
        results = []
        for boxes_, box_deltas_, intrinsics_ in zip(
            boxes, box_deltas, intrinsics
        ):
            if len(boxes_) == 0:
                results.append(Boxes3D.empty(boxes_.device))
                continue

            box_deltas_ = box_deltas_[
                torch.arange(box_deltas_.shape[0]), boxes_.class_ids
            ]

            # depth uncertainty
            depth_uncertainty = box_deltas_[:, -1].unsqueeze(-1)
            depth_uncertainty = depth_uncertainty.clamp(min=0.0, max=1.0)

            # center
            delta_center = box_deltas_[:, 0:2] * self.center_scale
            center_2d = boxes_.center + delta_center
            depth = torch.exp(box_deltas_[:, 2:3] / self.depth_log_scale)
            center_3d = unproject_points(center_2d, depth, intrinsics_)

            # dimensions
            dimensions = torch.exp(box_deltas_[:, 3:6] / self.dim_log_scale)

            # rot_y
            alpha = rotation_output_to_alpha(
                box_deltas_[:, 6:-1], self.num_rotation_bins
            )
            rot_y = alpha2yaw(alpha, center_3d)
            orientation = torch.stack(
                [torch.zeros_like(rot_y), rot_y, torch.zeros_like(rot_y)], -1
            )

            boxes3d = Boxes3D(
                torch.cat(
                    [center_3d, dimensions, orientation, depth_uncertainty], 1
                ),
                boxes_.class_ids,
            )
            results.append(boxes3d)

        return results
