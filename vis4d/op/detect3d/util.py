"""Utilitiy functions for detection 3D ops."""

from __future__ import annotations

import torch
from torch import Tensor

from vis4d.common.imports import VIS4D_CUDA_OPS_AVAILABLE

if VIS4D_CUDA_OPS_AVAILABLE:
    from vis4d_cuda_ops import nms_rotated  # pylint: disable=no-name-in-module


def bev_3d_nms(
    center_x: Tensor,
    center_y: Tensor,
    width: Tensor,
    length: Tensor,
    angle: Tensor,
    scores: Tensor,
    class_ids: Tensor | None = None,
    iou_threshold: float = 0.1,
) -> Tensor:
    """BEV 3D NMS.

    Args:
        center_x (Tensor): Center x of boxes. In shape (N, 1).
        center_y (Tensor): Center y of boxes. In shape (N, 1).
        width (Tensor): Width of boxes. In shape (N, 1).
        length (Tensor): Length of boxes. In shape (N, 1).
        angle (Tensor): Angle of boxes. In shape (N, 1).
        scores (Tensor): Scores of boxes. In shape (N, 1).
        class_ids (Tensor | None, optional): Class ids of boxes. In shape
            (N,). Defaults to None. If None, class_agnostic NMS will be
            performed.
        iou_threshold (float, optional): IoU threshold. Defaults to 0.1.

    Returns:
        Tensor: Indices of boxes that have been kept by NMS.
    """
    class_ids = (
        torch.zeros_like(scores, dtype=torch.int64)  # class_agnostic
        if class_ids is None
        else class_ids
    )

    return batched_nms_rotated(
        torch.cat([center_x, center_y, width, length, angle], dim=-1),
        scores,
        class_ids,
        iou_threshold,
    )


def batched_nms_rotated(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor): Boxes where NMS will be performed. They are expected to
            be in (x_ctr, y_ctr, width, height, angle_degrees) format. In shape
            (N, 5).
        scores (Tensor): Scores for each one of the boxes. In shape (N,).
        idxs (Tensor): Indices of the categories for each one of the boxes.
            In shape (N,).
        iou_threshold (float): Discards all overlapping boxes with IoU <
            iou_threshold.

    Returns:
        Tensor: Int64 tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores
    """
    assert boxes.shape[-1] == 5

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    boxes = boxes.float()  # fp16 does not have enough range for batched NMS

    # Strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap

    # Note that batched_nms in torchvision/ops/boxes.py only uses
    # max_coordinate, which won't handle negative coordinates correctly.
    # Here by using min_coordinate we can make sure the negative coordinates
    # are correctly handled.
    max_coordinate = (
        torch.max(boxes[:, 0], boxes[:, 1])
        + torch.max(boxes[:, 2], boxes[:, 3]) / 2
    ).max()
    min_coordinate = (
        torch.min(boxes[:, 0], boxes[:, 1])
        - torch.max(boxes[:, 2], boxes[:, 3]) / 2
    ).min()
    offsets = idxs.to(boxes) * (max_coordinate - min_coordinate + 1)
    boxes_for_nms = (
        boxes.clone()
    )  # avoid modifying the original values in boxes
    boxes_for_nms[:, :2] += offsets[:, None]

    if not VIS4D_CUDA_OPS_AVAILABLE:
        raise RuntimeError(
            "Please install vis4d_cuda_ops to use batched_nms_rotated"
        )
    keep = nms_rotated(  # pylint: disable=possibly-used-before-assignment
        boxes_for_nms, scores, iou_threshold
    )
    return keep
