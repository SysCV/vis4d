"""Utility functions for bounding boxes."""
from typing import Dict, List, Optional, Tuple

import torch
from torchvision.ops import batched_nms

from vis4d.op.geometry.transform import transform_points


def bbox_scale(
    boxes: torch.Tensor, scale_factor_xy: Tuple[float, float]
) -> torch.Tensor:
    """Scale bounding box tensor."""
    boxes[:, [0, 2]] *= scale_factor_xy[0]
    boxes[:, [1, 3]] *= scale_factor_xy[1]
    return boxes


def bbox_clip(
    boxes: torch.Tensor, image_hw: Tuple[float, float]
) -> torch.Tensor:
    """Clip bounding boxes to image dims."""
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, image_hw[1] - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, image_hw[0] - 1)
    return boxes


def scale_and_clip_boxes(
    boxes: torch.Tensor,
    original_hw: Tuple[int, int],
    output_hw: Tuple[int, int],
    clip: bool = True,
) -> torch.Tensor:
    """Postprocess boxes by scaling and clipping to given image dims."""
    scale_factor = (
        original_hw[1] / output_hw[1],
        original_hw[0] / output_hw[0],
    )
    boxes = bbox_scale(boxes, scale_factor)
    if clip:
        boxes = bbox_clip(boxes, original_hw)
    return boxes


@torch.jit.script
def bbox_area(boxes: torch.Tensor) -> torch.Tensor:
    """Compute bounding box areas.

    Args:
        boxes (torch.Tensor): [N, 4] tensor of 2D boxes in format (x1, y1, x2, y2).

    Returns:
        torch.Tensor: [N,] tensor of box areas.
    """
    return (boxes[:, 2] - boxes[:, 0]).clamp(0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(0)


@torch.jit.script
def bbox_intersection(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> torch.Tensor:
    """Given two lists of boxes of size N and M, compute N x M intersection.

    Args:
        boxes1: N 2D boxes in format (x1, y1, x2, y2)
        boxes2: M 2D boxes in format (x1, y1, x2, y2)

    Returns:
        Tensor: intersection (N, M).
    """
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )
    width_height.clamp_(min=0)
    intersection = width_height.prod(dim=2)
    return intersection


@torch.jit.script
def bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between all pairs of boxes.

    Args:
        boxes1: N 2D boxes in format (x1, y1, x2, y2)
        boxes2: M 2D boxes in format (x1, y1, x2, y2)

    Returns:
        Tensor: IoU (N, M).
    """
    area1 = bbox_area(boxes1)
    area2 = bbox_area(boxes2)
    inter = bbox_intersection(boxes1, boxes2)

    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def transform_bbox(
    trans_mat: torch.Tensor, boxes: torch.Tensor
) -> torch.Tensor:
    """Apply trans_mat (3, 3) / (B, 3, 3)  to (N, 4) / (B, N, 4) xyxy boxes."""
    assert len(trans_mat.shape) == len(
        boxes.shape
    ), "trans_mat and boxes must have same number of dimensions!"
    x1y1 = boxes[..., :2]
    x2y1 = torch.stack((boxes[..., 2], boxes[..., 1]), -1)
    x2y2 = boxes[..., 2:]
    x1y2 = torch.stack((boxes[..., 0], boxes[..., 3]), -1)

    x1y1 = transform_points(x1y1, trans_mat)
    x2y1 = transform_points(x2y1, trans_mat)
    x2y2 = transform_points(x2y2, trans_mat)
    x1y2 = transform_points(x1y2, trans_mat)

    x_all = torch.stack(
        (x1y1[..., 0], x2y2[..., 0], x2y1[..., 0], x1y2[..., 0]), -1
    )
    y_all = torch.stack(
        (x1y1[..., 1], x2y2[..., 1], x2y1[..., 1], x1y2[..., 1]), -1
    )
    transformed_boxes = torch.stack(
        (
            x_all.min(dim=-1)[0],
            y_all.min(dim=-1)[0],
            x_all.max(dim=-1)[0],
            y_all.max(dim=-1)[0],
        ),
        -1,
    )

    if len(boxes.shape) == 2:
        transformed_boxes.squeeze(0)
    return transformed_boxes


def random_choice(tensor: torch.Tensor, sample_size: int) -> torch.Tensor:
    """Randomly choose elements from a tensor."""
    perm = torch.randperm(len(tensor), device=tensor.device)[:sample_size]
    return tensor[perm]


def non_intersection(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Get the elements of t1 that are not present in t2."""
    compareview = t2.repeat(t1.shape[0], 1).T
    return t1[(compareview != t1).T.prod(1) == 1]


def apply_mask(
    masks: List[torch.Tensor], *args: List[torch.Tensor]
) -> Tuple[List[torch.Tensor], ...]:
    """Apply given masks (either bool or indices) to given list of tensors.

    Args:
        masks (List[torch.Tensor]): Masks to apply on tensors.
        *args (List[torch.Tensor]): List of tensors to apply the masks on.
    Returns:
        Tuple[List[torch.Tensor], ...]: Masked tensor lists.
    """
    return tuple(
        [t[m] if len(t) > 0 else t for t, m in zip(t_list, masks)]
        for t_list in args
    )


def filter_boxes_by_area(
    boxes: torch.Tensor, min_area: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Filter a set of 2D bounding boxes given a minimum area.
    Args:
        boxes (Tensor): 2D bounding boxes [N, 4].
        min_area (float, optional): Minimum area. Defaults to 0.0.

    Returns:
        Tuple[Tensor, Tensor]: filtered boxes, boolean mask
    """
    if min_area > 0.0:
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        valid_mask = w * h >= min_area
        if not valid_mask.all():
            return boxes[valid_mask], valid_mask
    return boxes, boxes.new_ones((len(boxes),), dtype=torch.bool)


def multiclass_nms(
    multi_bboxes,
    multi_scores,
    score_thr,
    iou_thr,
    max_num=-1,
    score_factors=None,
    return_inds=False,
):  # TODO copied from mmdet, revise
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.
    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4
        )

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes
        )
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "[ONNX Error] Can not record NMS "
                "as it has not been executed this time"
            )
        if return_inds:
            return bboxes, scores, labels, inds
        else:
            return bboxes, scores, labels

    keep = batched_nms(bboxes, scores, labels, iou_thr)

    if max_num > 0:
        keep = keep[:max_num]

    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if return_inds:
        return bboxes, scores, labels, inds[keep]
    else:
        return bboxes, scores, labels


# TODO revise
def distance_3d_nms(
    boxes3d,
    cat_mapping: Dict[int, str],
    boxes2d=None,
) -> torch.Tensor:
    """Distance based 3D NMS."""
    keep_indices = torch.ones(len(boxes3d)).bool()

    if boxes2d is not None:
        boxes2d_scores = boxes2d.score
    else:  # pragma: no cover
        boxes2d_scores = torch.ones(len(boxes3d))

    distance_matrix = torch.cdist(
        boxes3d.center.unsqueeze(0),
        boxes3d.center.unsqueeze(-1).transpose(1, 2),
    ).squeeze(-1)

    for i, box3d in enumerate(boxes3d):
        current_3d_score = box3d.score * boxes2d_scores[i]  # type: ignore
        current_class = cat_mapping[int(box3d.class_ids)]

        if current_class in ["pedestrian", "traffic_cone"]:
            nms_dist = 0.5
        elif current_class in ["bicycle", "motorcycle", "barrier"]:
            nms_dist = 1
        else:
            nms_dist = 2

        nms_candidates = (distance_matrix[i] < nms_dist).nonzero().squeeze(-1)

        valid_candidates = (
            boxes3d[nms_candidates].score * boxes2d_scores[nms_candidates]  # type: ignore # pylint: disable=line-too-long
            > current_3d_score
        )[(boxes3d[nms_candidates].class_ids == box3d.class_ids).squeeze(0)]

        if torch.any(valid_candidates):
            keep_indices[i] = False

    return keep_indices
