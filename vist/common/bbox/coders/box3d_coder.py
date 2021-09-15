import numpy as np
import torch
from typing import List
from vist.struct import Boxes2D, Boxes3D, Intrinsics
from ...geometry.projection import project_points, unproject_points
from ...geometry.rotation import yaw2alpha, get_alpha, gen_bin_rot
from .base import BaseBoxCoderConfig, BaseBoxCoder3D


class Box3DCoderConfig(BaseBoxCoderConfig):
    with_uncertainty: bool = True
    uncertainty_thres: float = 0.9
    depth_log_scale: float = 2.0
    dim_log_scale: float = 2.0


class CenterProjBox3DCoder(BaseBoxCoder3D):
    """3D bounding box coder based on center projection."""

    def __init__(self, cfg: Box3DCoderConfig) -> None:
        """Init."""
        self.cfg = Box3DCoderConfig(**cfg.dict())

    def encode(self, boxes: List[Boxes2D], targets: List[Boxes3D], intrinsics: Intrinsics) -> List[torch.Tensor]:
        """Encode deltas between boxes and targets given intrinsics."""
        result = []
        for boxes_, targets_, intrinsics_ in zip(boxes, targets, intrinsics):
            # delta center 2d
            projected_3d_center = project_points(targets_.center, intrinsics_)
            delta_center = projected_3d_center - boxes_.center

            # depth
            depth = torch.log(boxes_.center[:, -1]) * self.cfg.depth_log_scale
            depth = depth.unsqueeze(-1)

            # dimensions
            dims = torch.log(targets_.dimensions) * self.cfg.dim_log_scale

            # rotation
            alpha = yaw2alpha(targets_.rot_y, targets_.center)
            bin_cls = torch.zeros((alpha.shape[0], 2), device=alpha.device)
            bin_res = torch.zeros((alpha.shape[0], 2), device=alpha.device)
            for i in range(alpha.shape[0]):
                if alpha[i] < np.pi / 6.0 or alpha[i] > 5 * np.pi / 6.0:
                    bin_cls[i, 0] = 1
                    bin_res[i, 0] = alpha[i] - (-0.5 * np.pi)

                if alpha[i] > -np.pi / 6.0 or alpha[i] < -5 * np.pi / 6.0:
                    bin_cls[i, 1] = 1
                    bin_res[i, 1] = alpha[i] - (0.5 * np.pi)

            result.append(torch.cat(
            [delta_center, depth, dims, bin_cls, bin_res], -1
            ))

        return result

    def decode(self, boxes: List[Boxes2D], box_deltas: List[torch.Tensor], intrinsics: Intrinsics) -> List[Boxes3D]:
        """Decode the predicted box_deltas according to given base boxes."""
        results = []
        for boxes_, box_deltas_, intrinsics_ in zip(boxes, box_deltas, intrinsics):
            boxes_ = boxes_[torch.arange(box_deltas_.shape[0]), boxes_.class_ids]

            # depth uncertainty
            if self.cfg.with_uncertainty:
                depth_uncertainty = box_deltas_[:, 14:15]
            else:
                depth_uncertainty = torch.ones_like(box_deltas_[:, :1])

            depth_uncertainty = depth_uncertainty.clamp(min=0.0, max=1.0)

            # Depth filter
            keep = (depth_uncertainty > self.cfg.uncertainty_thres).view(
                box_deltas_.shape[0]
            )
            depth_uncertainty = depth_uncertainty[keep]

            # TODO do we really want to filter here? This is supposed to be a decoding function
            boxes_ = boxes_[keep]
            box_deltas_ = box_deltas_[keep]

            # center
            delta_center = box_deltas_[:, 0:2]
            center_2d = boxes_.center + delta_center
            depth = torch.exp(box_deltas_[:, 2:3] / self.cfg.depth_log_scale)
            center_3d = unproject_points(center_2d, depth, intrinsics_)

            # dimensions
            # TODO shouldn't this regress from the mean?
            dimensions = torch.exp(box_deltas_[:, 3:6] / self.cfg.dim_log_scale)

            # rotation
            rot = gen_bin_rot(box_deltas_[:, 6:14])

            # alpha2rot_y
            intrinsic_matrix = intrinsics_.tensor.squeeze(0)
            rot_y = get_alpha(rot) + torch.atan2(delta_center[..., 0] - intrinsic_matrix[0, 2], intrinsic_matrix[0, 0])
            rot_y = rot_y % (2 * np.pi) - np.pi
            rot_y = rot_y.unsqueeze(-1)

            boxes3d = Boxes3D(torch.cat([center_3d, dimensions, rot_y, depth_uncertainty], 1), boxes_.class_ids)
            results.append(boxes3d)

        return results
