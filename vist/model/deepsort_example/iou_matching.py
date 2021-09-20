"""IOU matching."""
from __future__ import absolute_import

from typing import Any, Dict, List, Optional

import torch

from vist.common.bbox.utils import compute_iou
from vist.struct.labels import Boxes2D

from .linear_assignment import INFTY_COST


def xyah_to_tlbr(bbox_xyah: torch.tensor) -> torch.tensor:
    """Convert a single one dimension tlbr box to xyah.

    Args:
        bbox_xyah: (4, ) [center_x, center_y, aspect_ratio, height]

    Returns:
        bbox_tlbr: shape (4,)
                    [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    """
    bbox_tlbr = bbox_xyah.clone().detach()
    height = bbox_xyah[3]
    half_width = (height * bbox_xyah[2]) / 2.0
    half_height = height / 2.0

    bbox_tlbr[0] = bbox_xyah[0] - half_width
    bbox_tlbr[1] = bbox_xyah[1] - half_height
    bbox_tlbr[2] = bbox_xyah[0] + half_width
    bbox_tlbr[3] = bbox_xyah[1] + half_height
    return bbox_tlbr


def iou_cost(
    tracks: Dict[int, Dict[str, Any]],
    detections: Boxes2D,
    track_ids: Optional[List[int]] = None,
    detection_indices: Optional[List[int]] = None,
):
    """An intersection over union distance metric.

    Args:
        tracks : A list of tracks.
        detections : A list of detections.
        track_ids : A list of track ids that should be matched.
            Defaults to all `tracks`.
        detection_indices : A list of indices to detections that should be
            matched. Defaults to all `detections`.

    Returns:
        cost_matrix: a cost matrix of shape
        [len(track_indices), len(detection_indices)], where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.
    """
    if track_ids is None:
        track_ids = torch.tensor(list(tracks.keys())).to(detections.device)
    if detection_indices is None:
        detection_indices = torch.arange(len(detections)).to(detections.device)

    cost_matrix = torch.zeros((len(track_ids), len(detection_indices)))
    for row, track_id in enumerate(track_ids):
        if tracks[track_id]["time_since_update"] > 1:
            cost_matrix[row, :] = INFTY_COST
            continue

        bbox = xyah_to_tlbr(tracks[track_id]["mean"][:4])
        conf = tracks[track_id]["confidence"]
        bbox = torch.cat(
            (bbox, torch.tensor([conf]).to(bbox.device))
        ).unsqueeze(0)
        box2d = Boxes2D(bbox)
        candidates = detections[detection_indices]

        iou_res = compute_iou(box2d, candidates).squeeze()

        cost_matrix[row, :] = torch.ones_like(iou_res) - iou_res
    return cost_matrix
