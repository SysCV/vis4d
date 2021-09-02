"""Utility functions for bounding boxes."""
import torch
import numpy as np

from detectron2.structures import Boxes, pairwise_iou

from vist.struct import Boxes2D, Boxes3D


def compute_iou(boxes1: Boxes2D, boxes2: Boxes2D) -> torch.Tensor:
    """Compute IoU between all pairs of boxes.

    Args:
        boxes1, boxes2 (Boxes2D): Contains N & M boxes.

    Returns:
        Tensor: IoU, size [N, M].
    """
    return pairwise_iou(Boxes(boxes1.boxes[:, :4]), Boxes(boxes2.boxes[:, :4]))


def random_choice(tensor: torch.Tensor, sample_size: int) -> torch.Tensor:
    """Randomly choose elements from a tensor."""
    perm = torch.randperm(len(tensor), device=tensor.device)[:sample_size]
    return tensor[perm]


def non_intersection(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Get the elements of t1 that are not present in t2."""
    compareview = t2.repeat(t1.shape[0], 1).T
    return t1[(compareview != t1).T.prod(1) == 1]


def project_points_to_image(
    points: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """Project Nx3 points to Nx2 pixel coordinates with 3x3 intrinsics."""
    hom_cam_coords = points / points[:, 2:3]
    pts_2d = hom_cam_coords.mm(intrinsics.t())
    return pts_2d[:, :2]  # type: ignore


def yaw2alpha(rot_y: torch.tensor, x_loc: torch.tensor, z_loc: torch.tensor):
    """
    Get alpha by rotation_y - theta.

    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    alpha : Observation angle of object, ranging [-pi..pi]
    """
    torch_pi = rot_y.new_tensor([np.pi])
    alpha = rot_y - torch.atan2(x_loc, z_loc)
    alpha = (alpha + torch_pi) % (2 * torch_pi) - torch_pi
    return alpha


def get_alpha(rot):
    """Get alpha from rotation y and bins."""
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    idx = (rot[:, 1] > rot[:, 5]).float()
    alpha1 = torch.atan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = torch.atan(rot[:, 6] / rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


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
        bbox: torch.tensor,
        pred_bboxes: torch.tensor,
        cam_intrinsics: torch.tensor,
    ):
        """Decode the model prediction."""
        # center 2d
        delta_center = pred_bboxes[:, 0:2]
        delta_center = delta_center * stds + means

        pred_x = (proposals[:, 0] + proposals[:, 2]) * 0.5
        pred_y = (proposals[:, 1] + proposals[:, 3]) * 0.5

        box_cen = torch.cat([pred_x, pred_2d], 1)

        cen2d_pred = box_cen + delta_center

        # depth
        depth = torch.exp(pred_bboxes[:, 2] / self.depth_log_scale)

        # dimensions
        dimensions = torch.exp(pred_bboxes[:, 3:6] / self.dim_log_scale)

        # rot_y
        orientation = pred_bboxes[:, 6:14]

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

        # depth uncertainty
        if pred_bboxes.shape[-1] > 14:  # if with confidence
            depth_uncertainty = pred_bboxes[:, 14:15]
        else:
            depth_uncertainty = torch.ones_like(pred_bboxes[:, :1])

        # center 3d
        center = (
            torch.cat([cen2d_pred, torch.ones_like(cen2d_pred)[..., 0:1]], -1)
            @ torch.inverse(cam_intrinsics).T
        )
        center *= depth.unsqueeze(-1)

        return torch.cat(
            [depth_uncertainty, center, dimensions, rot_y.unsqueeze(-1)], -1
        )
