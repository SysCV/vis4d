"""Utility functions for bounding boxes."""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.ops import batched_nms, nms

from vis4d.common.logging import rank_zero_warn
from vis4d.op.geometry.transform import transform_points


def bbox_scale(
    boxes: torch.Tensor, scale_factor_xy: tuple[float, float]
) -> torch.Tensor:
    """Scale bounding box tensor.

    Args:
        boxes (torch.Tensor): Bounding boxes with shape [N, 4]
        scale_factor_xy (tuple[float, float]): Scaling factor for x and y

    Returns:
        torch.Tensor with bounding boxes scaled by the given factors in
        x and y direction
    """
    boxes[:, [0, 2]] *= scale_factor_xy[0]
    boxes[:, [1, 3]] *= scale_factor_xy[1]
    return boxes


def bbox_clip(
    boxes: torch.Tensor,
    image_hw: tuple[float, float],
    epsilon: int = 0,
) -> torch.Tensor:
    """Clip bounding boxes to image dims.

    Args:
        boxes (torch.Tensor): Bounding boxes with shape [N, 4]
        image_hw (tuple[float, float]): Image dimensions.
        epsilon (int): Epsilon for clipping.
            Defaults to 0.

    Returns:
        torch.Tensor: Clipped bounding boxes.
    """
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, image_hw[1] - epsilon)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, image_hw[0] - epsilon)
    return boxes


def scale_and_clip_boxes(
    boxes: torch.Tensor,
    original_hw: tuple[int, int],
    current_hw: tuple[int, int],
    clip: bool = True,
) -> torch.Tensor:
    """Postprocess boxes by scaling and clipping to given image dims.

    Args:
        boxes (torch.Tensor): Bounding boxes with shape [N, 4].
        original_hw (tuple[int, int]): Original height / width of image.
        current_hw (tuple[int, int]): Current height / width of image.
        clip (bool): If true, clips box corners to image bounds.

    Returns:
        torch.Tensor: Rescaled and possibly clipped bounding boxes.
    """
    scale_factor = (
        original_hw[1] / current_hw[1],
        original_hw[0] / current_hw[0],
    )
    boxes = bbox_scale(boxes, scale_factor)
    if clip:
        boxes = bbox_clip(boxes, original_hw)
    return boxes


def bbox_area(boxes: torch.Tensor) -> torch.Tensor:
    """Compute bounding box areas.

    Args:
        boxes (torch.Tensor): [N, 4] tensor of 2D boxes
                                     in format (x1, y1, x2, y2).

    Returns:
        torch.Tensor: [N,] tensor of box areas.
    """
    return (boxes[:, 2] - boxes[:, 0]).clamp(0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(0)


def bbox_intersection(boxes1: Tensor, boxes2: Tensor) -> torch.Tensor:
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

    union = area1[:, None] + area2 - inter

    inter = torch.where(
        union > 0,
        inter,
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )

    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def bbox_intersection_aligned(boxes1: Tensor, boxes2: Tensor) -> torch.Tensor:
    """Given two lists of boxes both of size N, compute N intersection.

    Args:
        boxes1: N 2D boxes in format (x1, y1, x2, y2)
        boxes2: N 2D boxes in format (x1, y1, x2, y2)

    Returns:
        Tensor: intersection (N).
    """
    width_height = torch.min(boxes1[:, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, :2], boxes2[:, :2]
    )
    width_height.clamp_(min=0)
    intersection = width_height.prod(dim=1)
    return intersection


def bbox_iou_aligned(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> torch.Tensor:
    """Compute IoU between aligned pairs of boxes.

    The number of boxes in both inputs must be the same.

    Args:
        boxes1: N 2D boxes in format (x1, y1, x2, y2)
        boxes2: N 2D boxes in format (x1, y1, x2, y2)

    Returns:
        Tensor: IoU (N).
    """
    area1 = bbox_area(boxes1)
    area2 = bbox_area(boxes2)
    inter = bbox_intersection_aligned(boxes1, boxes2)

    iou = torch.where(
        inter > 0,
        inter / (area1 + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def transform_bbox(
    trans_mat: torch.Tensor, boxes: torch.Tensor
) -> torch.Tensor:
    """Apply trans_mat (3, 3) / (B, 3, 3)  to (N, 4) / (B, N, 4) xyxy boxes.

    Args:
        trans_mat (torch.Tensor): Transformation matrix
                                  of shape (3,3) or (B,3,3)
        boxes (torch.Tensor): Bounding boxes of shape (N,4) or (B,N,4)

    Returns:
        torch.Tensor containing linear transformed bounding boxes. (B?, N, 4)
    """
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


# TODO, refactor? move to utils?
def random_choice(tensor: torch.Tensor, sample_size: int) -> torch.Tensor:
    """Randomly choose elements from a tensor.

    If sample_size < len(tensor) this function will sample without repetition
    otherwise certain elements will be repeated.

    Args:
        tensor (torch.Tensor): Tensor to sample from
        sample_size (int): Number of elements to sample

    Returns:
        torch.Tensor containing sample_size randomly sampled entries.
    """
    perm = torch.randperm(len(tensor), device=tensor.device)[:sample_size]

    # Additionally sample with repetition
    if sample_size > len(tensor):
        remaining_samples = sample_size - len(tensor)
        perm = torch.concat(
            [
                torch.randint(
                    remaining_samples,
                    (remaining_samples,),
                    device=tensor.device,
                ),
                perm,
            ]
        )

    return tensor[perm]


def non_intersection(
    tensor_a: torch.Tensor, tensor_b: torch.Tensor
) -> torch.Tensor:
    """Get the elements of tensor_a that are not present in tensor_b.

    Args:
        tensor_a (torch.Tensor): First tensor
        tensor_b (torch.Tensor): Second tensor

    Returns:
        torch.Tensor containing all elements that occur in both tensors
    """
    compareview = tensor_b.repeat(tensor_a.shape[0], 1).T
    return tensor_a[(compareview != tensor_a).T.prod(1) == 1]


def apply_mask(
    masks: list[torch.Tensor], *args: list[torch.Tensor]
) -> tuple[list[torch.Tensor], ...]:
    """Apply given masks (either bool or indices) to given list of tensors.

    Args:
        masks (list[torch.Tensor]): Masks to apply on tensors.
        *args (list[torch.Tensor]): List of tensors to apply the masks on.

    Returns:
        tuple[list[torch.Tensor], ...]: Masked tensor lists.
    """
    return tuple(
        [t[m] if len(t) > 0 else t for t, m in zip(t_list, masks)]
        for t_list in args
    )


def filter_boxes_by_area(
    boxes: torch.Tensor, min_area: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter a set of 2D bounding boxes given a minimum area.

    Args:
        boxes (Tensor): 2D bounding boxes [N, 4].
        min_area (float, optional): Minimum area. Defaults to 0.0.

    Returns:
        tuple[Tensor, Tensor]: filtered boxes, boolean mask
    """
    if min_area > 0.0:
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        valid_mask = w * h >= min_area
        if not valid_mask.all():
            return boxes[valid_mask], valid_mask
    return boxes, boxes.new_ones((len(boxes),), dtype=torch.bool)


def hbox2corner(boxes: Tensor) -> Tensor:
    """Convert box coordinates from boxes to corners.

    Boxes are represented as (x1, y1, x2, y2).
    Corners are represented as ((x1, y1), (x2, y1), (x1, y2), (x2, y2)).

    Args:
        boxes (Tensor): Horizontal box tensor with shape of (..., 4).

    Returns:
        Tensor: Corner tensor with shape of (..., 4, 2).
    """
    x1, y1, x2, y2 = torch.split(boxes, 1, dim=-1)
    corners = torch.cat([x1, y1, x2, y1, x1, y2, x2, y2], dim=-1)
    return corners.reshape(*corners.shape[:-1], 4, 2)


def corner2hbox(corners: Tensor) -> Tensor:
    """Convert box coordinates from corners to boxes.

    Boxes are represented as (x1, y1, x2, y2).
    Corners are represented as ((x1, y1), (x2, y1), (x1, y2), (x2, y2)).

    Args:
        corners (Tensor): Corner tensor with shape of (..., 4, 2).

    Returns:
        Tensor: Horizontal box tensor with shape of (..., 4).
    """
    if corners.numel() == 0:
        return corners.new_zeros((0, 4))
    min_xy = corners.min(dim=-2)[0]
    max_xy = corners.max(dim=-2)[0]
    return torch.cat([min_xy, max_xy], dim=-1)


def bbox_project(boxes: Tensor, homography_matrix: Tensor) -> Tensor:
    """Apply geometric transform to boxes in-place.

    Args:
        boxes (Tensor): Horizontal box tensor with shape of (..., 4).
        homography_matrix (Tensor): Shape (3, 3) for geometric transformation.
    """
    corners = hbox2corner(boxes)
    corners = torch.cat(
        [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1
    )
    corners_t = torch.transpose(corners, -1, -2)
    corners_t = torch.matmul(homography_matrix, corners_t)
    corners = torch.transpose(corners_t, -1, -2)
    # Convert to homogeneous coordinates by normalization
    corners = corners[..., :2] / corners[..., 2:3]
    return corner2hbox(corners)


def multiclass_nms(
    multi_bboxes: Tensor,
    multi_scores: Tensor,
    score_thr: float,
    iou_thr: float,
    max_num: int = -1,
    class_agnostic: bool = False,
    split_thr: int = 100000,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Non-maximum suppression with multiple classes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Defaults to -1.
        class_agnostic (bool, optional): whether apply class_agnostic NMS.
            Defaults to False.
        split_thr (int, optional): If the number of bboxes is less than
            split_thr, use class agnostic NMS with class_agnostic=True.
            Defaults to 100000.

    Returns:
        tuple: (Tensor, Tensor, Tensor, Tensor): detections (k, 5), scores
            (k), classes (k) and indices (k).

    Raises:
        RuntimeError: If there is a onnx error,
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
        # NonZero not supported in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported in TensorRT
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
        return bboxes, scores, labels, inds

    if class_agnostic and bboxes.shape[0] < split_thr:
        keep = nms(bboxes, scores, iou_thr)
    else:
        if class_agnostic:
            rank_zero_warn(
                f"Number of bboxes is larger than {split_thr}, "
                "using per-class NMS instead"
            )
        keep = batched_nms(bboxes, scores, labels, iou_thr)

    if max_num > 0:
        keep = keep[:max_num]

    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    return bboxes, scores, labels, inds[keep]
