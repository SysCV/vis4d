"""XYWH Delta coder for 2D boxes.

Modified from mmdetection (https://github.com/open-mmlab/mmdetection).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


class DeltaXYWHBBoxEncoder:
    """Delta XYWH BBox encoder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    it encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh).
    """

    def __init__(
        self,
        target_means: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        target_stds: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        """Creates an instance of the class.

        Args:
            target_means (tuple, optional): Denormalizing means of target for
                delta coordinates. Defaults to (0.0, 0.0, 0.0, 0.0).
            target_stds (tuple, optional): Denormalizing standard deviation of
                target for delta coordinates. Defaults to (1.0, 1.0, 1.0, 1.0).
        """
        self.means = target_means
        self.stds = target_stds

    def __call__(self, boxes: Tensor, targets: Tensor) -> Tensor:
        """Get box regression transformation deltas.

        Used to transform target boxes into target regression parameters.

        Args:
            boxes (Tensor): Source boxes, e.g., object proposals.
            targets (Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            Tensor: Box transformation deltas
        """
        assert boxes.size(0) == targets.size(0)
        assert boxes.size(-1) == targets.size(-1) == 4
        encoded_bboxes = bbox2delta(boxes, targets, self.means, self.stds)
        return encoded_bboxes


class DeltaXYWHBBoxDecoder:
    """Delta XYWH BBox decoder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    it decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).
    """

    def __init__(
        self,
        target_means: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        target_stds: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        wh_ratio_clip: float = 16 / 1000,
    ) -> None:
        """Creates an instance of the class.

        Args:
            target_means (tuple, optional): Denormalizing means of target for
                delta coordinates. Defaults to (0.0, 0.0, 0.0, 0.0).
            target_stds (tuple, optional): Denormalizing standard deviation of
                target for delta coordinates. Defaults to (1.0, 1.0, 1.0, 1.0).
            wh_ratio_clip (float, optional): Maximum aspect ratio for boxes.
                Defaults to 16/1000.
        """
        self.means = target_means
        self.stds = target_stds
        self.wh_ratio_clip = wh_ratio_clip

    def __call__(self, boxes: Tensor, box_deltas: Tensor) -> Tensor:
        """Apply box offset energies box_deltas to boxes.

        Args:
            boxes (Tensor): Basic boxes. Shape (B, N, 4) or (N, 4)
            box_deltas (Tensor): Encoded offsets with respect to each roi.
               Has shape (B, N, num_classes * 4) or (B, N, 4) or
               (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
               when rois is a grid of anchors.Offset encoding follows [1]_.

        Returns:
            Tensor: Decoded boxes.
        """
        assert box_deltas.size(0) == boxes.size(0)
        decoded_boxes = delta2bbox(
            boxes, box_deltas, self.means, self.stds, self.wh_ratio_clip
        )
        return decoded_boxes


def bbox2delta(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    means: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    stds: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> Tensor:
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth boxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4).
        gt_boxes (Tensor): Gt boxes to be used as base, shape (N, ..., 4).
        means (Sequence[float]): Denormalizing means for delta coordinates.
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates.

    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size() == gt_boxes.size()

    proposals = proposals.float()
    gt = gt_boxes.float()
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

    mean_tensor = torch.tensor(means, dtype=deltas.dtype, device=deltas.device)
    std_tensor = torch.tensor(stds, dtype=deltas.dtype, device=deltas.device)
    deltas = deltas.sub_(mean_tensor.view(1, -1)).div_(std_tensor.view(1, -1))

    return deltas


def delta2bbox(
    rois: torch.Tensor,
    deltas: torch.Tensor,
    means: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    stds: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    wh_ratio_clip: float = 16 / 1000,
) -> Tensor:
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4).
        deltas (Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 4) or (N, 4). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1.).
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.

    Returns:
        Tensor: Boxes with shape (N, num_classes * 4) or (N, 4), where 4
           represent tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524
    """
    num_boxes, num_classes = deltas.size(0), deltas.size(1) // 4
    if num_boxes == 0:
        return deltas

    deltas = deltas.reshape(-1, 4)

    mean_tensor = torch.tensor(means, dtype=deltas.dtype, device=deltas.device)
    std_tensor = torch.tensor(stds, dtype=deltas.dtype, device=deltas.device)
    denorm_deltas = deltas * std_tensor.view(1, -1) + mean_tensor.view(1, -1)

    dxy = denorm_deltas[:, :2]
    dwh = denorm_deltas[:, 2:]

    # Compute width/height of each roi
    rois_ = rois.repeat(1, num_classes).reshape(-1, 4)
    pxy = (rois_[:, :2] + rois_[:, 2:]) * 0.5
    pwh = rois_[:, 2:] - rois_[:, :2]

    dxy_wh = pwh * dxy

    max_ratio = abs(math.log(wh_ratio_clip))
    dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    gxy = pxy + dxy_wh
    gwh = pwh * dwh.exp()
    x1y1 = gxy - (gwh * 0.5)
    x2y2 = gxy + (gwh * 0.5)
    boxes = torch.cat([x1y1, x2y2], dim=-1)
    boxes = boxes.reshape(num_boxes, -1)
    return boxes
