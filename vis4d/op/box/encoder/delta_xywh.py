"""XYWH Delta coder for 2D boxes.

Modified from mmdetection (https://github.com/open-mmlab/mmdetection).
"""
import math
from typing import Optional, Tuple

import torch

from .base import BoxEncoder2D


class DeltaXYWHBBoxEncoder(BoxEncoder2D):
    """Delta XYWH BBox encoder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    it encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and decodes
    delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).

    Args:
        target_means (Tuple[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Tuple[float]): Denormalizing standard deviation of
            target for delta coordinates
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(
        self,
        target_means: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        target_stds: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        clip_border: bool = True,
    ):
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border

    def encode(
        self, boxes: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            boxes (torch.Tensor): Source boxes, e.g., object proposals.
            targets (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert boxes.size(0) == targets.size(0)
        assert boxes.size(-1) == targets.size(-1) == 4
        encoded_bboxes = bbox2delta(boxes, targets, self.means, self.stds)
        return encoded_bboxes

    def decode(
        self,
        bboxes: torch.Tensor,
        pred_bboxes: torch.Tensor,
        max_shape: Optional[Tuple[int, int]] = None,
        wh_ratio_clip: float = 16 / 1000,
    ):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (torch.Tensor): Basic boxes. Shape (B, N, 4) or (N, 4)
            pred_bboxes (Tensor): Encoded offsets with respect to each roi.
               Has shape (B, N, num_classes * 4) or (B, N, 4) or
               (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
               when rois is a grid of anchors.Offset encoding follows [1]_.
            max_shape (Tuple[int, int]): Maximum bounds for boxes, specified
               as (H, W). Defaults to None.
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2bbox(
            bboxes,
            pred_bboxes,
            self.means,
            self.stds,
            max_shape,
            wh_ratio_clip,
            self.clip_border,
        )

        return decoded_bboxes


@torch.jit.script
def bbox2delta(
    proposals: torch.Tensor,
    gt: torch.Tensor,
    means: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    stds: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
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

    means = torch.tensor(means, dtype=deltas.dtype, device=deltas.device)
    stds = torch.tensor(stds, dtype=deltas.dtype, device=deltas.device)
    deltas = deltas.sub_(means.view(1, -1)).div_(stds.view(1, -1))

    return deltas


@torch.jit.script
def delta2bbox(
    rois: torch.Tensor,
    deltas: torch.Tensor,
    means: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    stds: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    max_shape: Optional[Tuple[int, int]] = None,
    wh_ratio_clip: float = 16 / 1000,
    clip_border: bool = True,
):
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
        max_shape (tuple[int, int]): Maximum bounds for boxes, specifies
           (H, W). Default None.
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Default True.

    Returns:
        Tensor: Boxes with shape (N, num_classes * 4) or (N, 4), where 4
           represent tl_x, tl_y, br_x, br_y.

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
    num_bboxes, num_classes = deltas.size(0), deltas.size(1) // 4
    if num_bboxes == 0:
        return deltas

    deltas = deltas.reshape(-1, 4)

    means = torch.tensor(means, dtype=deltas.dtype, device=deltas.device)
    stds = torch.tensor(stds, dtype=deltas.dtype, device=deltas.device)
    denorm_deltas = deltas * stds.view(1, -1) + means.view(1, -1)

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
    bboxes = torch.cat([x1y1, x2y2], dim=-1)
    if clip_border and max_shape is not None:
        bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
        bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])
    bboxes = bboxes.reshape(num_bboxes, -1)
    return bboxes