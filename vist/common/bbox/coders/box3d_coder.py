import numpy as np
import torch

from vist.common.bbox.utils import project_points_to_image, yaw2alpha, \
    get_alpha
from vist.struct import Boxes2D, Boxes3D


class Box3DCoderConfig():
    with_uncertainty: bool
    uncertainty_thres: float = 0.9


class Box3DCoder:
    """3D bounding box coder for QD-3DT."""

    def __init__(
        self,
        dep_log_scale: float = 2.0,
        dim_log_scale: float = 2.0,
    ):
        self.depth_log_scale = dep_log_scale
        self.dim_log_scale = dim_log_scale

    def encode(
        self,
        gt_bboxes_2d: Boxes2D,
        gt_bboxes_3d: Boxes3D,
        cam_intrinsics: torch.tensor,
    ):
        """Encode the GT into model prediction format for computing loss."""
        # delta center 2d
        projected_2dc = project_points_to_image(
            gt_bboxes_3d.boxes[:, :3], cam_intrinsics
        ).view(gt_bboxes_3d.boxes.shape[0], -1)

        x_2d = (gt_bboxes_2d.boxes[:, 0] + gt_bboxes_2d.boxes[:, 2]) / 2
        x_2d = x_2d.view(gt_bboxes_3d.boxes.shape[0], -1)

        y_2d = (gt_bboxes_2d.boxes[:, 1] + gt_bboxes_2d.boxes[:, 3]) / 2
        y_2d = y_2d.view(gt_bboxes_3d.boxes.shape[0], -1)

        center_2dc = torch.cat([x_2d, y_2d], 1)

        delta_center = projected_2dc - center_2dc
        delta_center = delta_center.view(gt_bboxes_3d.boxes.shape[0], -1)

        # depth
        depth = torch.log(gt_bboxes_3d.boxes[:, 2]) * self.depth_log_scale
        depth = depth.view(gt_bboxes_3d.boxes.shape[0], -1)

        # dimensions
        dimensions = torch.log(gt_bboxes_3d.boxes[:, 3:6]) * self.dim_log_scale
        dimensions = dimensions.view(gt_bboxes_3d.boxes.shape[0], -1)

        # roty to bins
        rot_y = gt_bboxes_3d.boxes[:, 6].view(gt_bboxes_3d.boxes.shape[0], -1)

        # alpha
        alpha = yaw2alpha(
            rot_y, gt_bboxes_3d.boxes[:, 0:1], gt_bboxes_3d.boxes[:, 2:3]
        )

        alpha = alpha % (2 * np.pi) - np.pi
        bin_cls = torch.zeros((alpha.shape[0], 2), device=alpha.device)
        bin_res = torch.zeros((alpha.shape[0], 2), device=alpha.device)
        for i in range(alpha.shape[0]):
            if alpha[i] < np.pi / 6.0 or alpha[i] > 5 * np.pi / 6.0:
                bin_cls[i, 0] = 1
                bin_res[i, 0] = alpha[i] - (-0.5 * np.pi)

            if alpha[i] > -np.pi / 6.0 or alpha[i] < -5 * np.pi / 6.0:
                bin_cls[i, 1] = 1
                bin_res[i, 1] = alpha[i] - (0.5 * np.pi)

        return torch.cat(
            [delta_center, depth, dimensions, bin_cls, bin_res], -1
        )

    def decode(
        self,
        bbox_2d_preds: Boxes2D,
        bbox_3d_preds: torch.Tensor,
        cam_intrinsics: torch.Tensor,
    ):
        """Decode the model prediction."""
        bbox_3d_preds = bbox_3d_preds[
            torch.arange(bbox_3d_preds.shape[0]), bbox_2d_preds.class_ids
        ]

        # depth uncertainty
        if with_uncertainty:
            depth_uncertainty = bbox_3d_preds[:, 14:15]
        else:
            depth_uncertainty = torch.ones_like(bbox_3d_preds[:, :1])

        depth_uncertainty = torch.clamp(depth_uncertainty, min=0.0, max=1.0)

        # Depth filter
        keep = (depth_uncertainty > uncertainty_thres).view(
            bbox_3d_preds.shape[0]
        )
        depth_uncertainty = depth_uncertainty[keep]

        bbox_2d_preds = bbox_2d_preds[keep]
        bbox_3d_preds = bbox_3d_preds[keep]

        # center 2d
        delta_center = bbox_3d_preds[:, 0:2]

        pred_x = (bbox_2d_preds.boxes[:, 0] + bbox_2d_preds.boxes[:, 2]) * 0.5
        pred_y = (bbox_2d_preds.boxes[:, 1] + bbox_2d_preds.boxes[:, 3]) * 0.5

        pred_x = pred_x.view(bbox_3d_preds.shape[0], 1)
        pred_y = pred_y.view(bbox_3d_preds.shape[0], 1)

        box_cen = torch.cat([pred_x, pred_y], 1)

        cen2d_pred = box_cen + delta_center

        # depth
        depth = torch.exp(bbox_3d_preds[:, 2] / self.depth_log_scale)
        depth = depth.view(bbox_3d_preds.shape[0], 1)

        # dimensions
        dimensions = torch.exp(bbox_3d_preds[:, 3:6] / self.dim_log_scale)

        # rotation
        orientation = bbox_3d_preds[:, 6:14]

        # bin 1
        divider1 = torch.sqrt(
            orientation[:, 2:3] ** 2 + orientation[:, 3:4] ** 2
        )
        b1sin = orientation[:, 2:3] / divider1
        b1cos = orientation[:, 3:4] / divider1

        # bin 2
        divider2 = torch.sqrt(
            orientation[:, 6:7] ** 2 + orientation[:, 7:8] ** 2
        )
        b2sin = orientation[:, 6:7] / divider2
        b2cos = orientation[:, 7:8] / divider2

        rot = torch.cat(
            [
                orientation[:, 0:2],
                b1sin,
                b1cos,
                orientation[:, 4:6],
                b2sin,
                b2cos,
            ],
            1,
        )

        # alpha2rot_y
        rot_y = get_alpha(rot) + torch.atan2(
            delta_center[..., 0] - cam_intrinsics[0, 2], cam_intrinsics[0, 0]
        )
        rot_y = rot_y % (2 * np.pi) - np.pi
        rot_y = rot_y.view(bbox_3d_preds.shape[0], 1)

        # center 3d
        center = (
            torch.cat([cen2d_pred, torch.ones_like(cen2d_pred)[..., 0:1]], -1)
            @ torch.inverse(cam_intrinsics).T
        )
        center *= depth

        bbox_3d_preds = Boxes3D(
            torch.cat([center, dimensions, rot_y, depth_uncertainty], 1),
            bbox_2d_preds.class_ids,
        )

        return bbox_2d_preds, bbox_3d_preds