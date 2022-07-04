"""Utility functions for bounding boxes."""
from typing import Dict, Optional

import torch



def bbox_intersection(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
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


def bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between all pairs of boxes.

    Args:
        boxes1: N 2D boxes in format (x1, y1, x2, y2, Optional[score])
        boxes2: M 2D boxes in format (x1, y1, x2, y2, Optional[score])

    Returns:
        Tensor: IoU (N, M).
    """
    area1 = boxes1.area  # TODO revise function
    area2 = boxes2.area
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


def distance_3d_nms(
    boxes3d: torch.Tensor,
    cat_mapping: Dict[int, str],
) -> torch.Tensor:
    """Distance based 3D NMS.

    Args:

    """
    # TODO documentation, revise function
    keep_indices = torch.ones(len(boxes3d)).bool()

    distance_matrix = torch.cdist(
        boxes3d.center.unsqueeze(0),
        boxes3d.center.unsqueeze(-1).transpose(1, 2),
    ).squeeze(-1)

    for i, box3d in enumerate(boxes3d):
        current_class = cat_mapping[int(box3d.class_ids)]

        if current_class in ["pedestrian", "traffic_cone"]:
            nms_dist = 0.5
        elif current_class in ["bicycle", "motorcycle", "barrier"]:
            nms_dist = 1
        else:
            nms_dist = 2

        nms_candidates = (distance_matrix[i] < nms_dist).nonzero().squeeze(-1)

        valid_candidates = (
            boxes3d[nms_candidates].score * boxes2d_scores[nms_candidates]
            > current_3d_score
        )[(boxes3d[nms_candidates].class_ids == box3d.class_ids).squeeze(0)]

        if torch.any(valid_candidates):
            keep_indices[i] = False

    return keep_indices
