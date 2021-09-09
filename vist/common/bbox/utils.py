"""Utility functions for bounding boxes."""
import torch
import numpy as np


from vist.struct import Boxes2D, Boxes3D
import pdb


def bbox_intersection(boxes1: Boxes2D, boxes2: Boxes2D) -> torch.Tensor:
    """Given two lists of boxes of size N and M, compute N x M intersection.

    Args:
        boxes1: N 2D boxes in format (x1, y1, x2, y2, Optional[score])
        boxes2: M 2D boxes in format (x1, y1, x2, y2, Optional[score])

    Returns:
        Tensor: intersection (N, M).
    """
    boxes1, boxes2 = boxes1.boxes[:, :4], boxes2.boxes[:, :4]
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )

    width_height.clamp_(min=0)
    intersection = width_height.prod(dim=2)
    return intersection


def bbox_iou(boxes1: Boxes2D, boxes2: Boxes2D) -> torch.Tensor:
    """Compute IoU between all pairs of boxes.

    Args:
        boxes1: N 2D boxes in format (x1, y1, x2, y2, Optional[score])
        boxes2: M 2D boxes in format (x1, y1, x2, y2, Optional[score])

    Returns:
        Tensor: IoU (N, M).
    """
    area1 = boxes1.area()
    area2 = boxes2.area()
    inter = bbox_intersection(boxes1, boxes2)

    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


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


def bbox2delta(
    proposals, gt, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0)
):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 4)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(
    rois,
    deltas,
    means=(0.0, 0.0, 0.0, 0.0),
    stds=(1.0, 1.0, 1.0, 1.0),
    max_shape=None,
    wh_ratio_clip=16 / 1000,
    clip_border=True,
    add_ctr_clamp=False,
    ctr_clamp=32,
):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4) or (B, N, 4)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (B, N, num_classes * 4) or (B, N, 4) or
            (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
            when rois is a grid of anchors.Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If rois shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.
        wh_ratio_clip (float): Maximum aspect ratio for boxes.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.

    Returns:
        Tensor: Boxes with shape (B, N, num_classes * 4) or (B, N, 4) or
           (N, num_classes * 4) or (N, 4), where 4 represent
           tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> delta2bbox(rois, deltas, max_shape=(32, 32, 3))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.1409, 0.1409, 2.8591, 2.8591],
                [0.0000, 0.3161, 4.1945, 0.6839],
                [5.0000, 5.0000, 5.0000, 5.0000]])
    """
    means = (
        deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(-1) // 4)
    )
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(-1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[..., 0::4]
    dy = denorm_deltas[..., 1::4]
    dw = denorm_deltas[..., 2::4]
    dh = denorm_deltas[..., 3::4]

    x1, y1 = rois[..., 0], rois[..., 1]
    x2, y2 = rois[..., 2], rois[..., 3]
    # Compute center of each roi
    px = ((x1 + x2) * 0.5).unsqueeze(-1).expand_as(dx)
    py = ((y1 + y2) * 0.5).unsqueeze(-1).expand_as(dy)
    # Compute width/height of each roi
    pw = (x2 - x1).unsqueeze(-1).expand_as(dw)
    ph = (y2 - y1).unsqueeze(-1).expand_as(dh)

    dx_width = pw * dx
    dy_height = ph * dy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dx_width = torch.clamp(dx_width, max=ctr_clamp, min=-ctr_clamp)
        dy_height = torch.clamp(dy_height, max=ctr_clamp, min=-ctr_clamp)
        dw = torch.clamp(dw, max=max_ratio)
        dh = torch.clamp(dh, max=max_ratio)
    else:
        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + dx_width
    gy = py + dy_height
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5

    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view(deltas.size())

    if clip_border and max_shape is not None:
        # clip bboxes with dynamic `min` and `max` for onnx
        if torch.onnx.is_in_onnx_export():
            from mmdet.core.export import dynamic_clip_for_onnx

            x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape)
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view(deltas.size())
            return bboxes
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = (
            torch.cat([max_shape] * (deltas.size(-1) // 2), dim=-1)
            .flip(-1)
            .unsqueeze(-2)
        )
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes


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

    # TODO: adjust for inference
    def decode(
        self,
        bbox_2d_preds: torch.tensor,
        bbox_3d_preds: torch.tensor,
    ):
        """Decode the model prediction."""
        pdb.set_trace()
        if bbox_2d_preds is not None:
            bbox_2d_preds = delta2bbox(
                rois[:, 1:],
                bbox_2d_preds,
                self.cfg.target_means,
                self.cfg.target_stds,
                img_shape,
            )
        else:
            bbox_2d_preds = rois[:, 1:].clone()
            if img_shape is not None:
                bbox_2d_preds[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bbox_2d_preds[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        # center 2d
        delta_center = bboxes_2d[:, 0:2]

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
