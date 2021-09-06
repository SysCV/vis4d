"""IOU matching."""
from __future__ import absolute_import

from typing import List, Optional

import torch

from .detection import Detection
from .linear_assignment import INFTY_COST
from .track import Track


def iou(bbox: torch.tensor, candidates: torch.tensor) -> torch.tensor:
    """Computer intersection over union.

    Args:
        bbox :
            A bounding box in format `(top left x, top left y, width, height)`.
        candidates :
            A matrix of candidate bounding boxes (one per row) in the same
            format as `bbox`.

    Returns:
        iou_res: The intersection over union in [0, 1] between the `bbox` and
            each candidate. A higher score means a larger fraction of the
            `bbox` is occluded by the candidate.
    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = torch.stack(
        [
            torch.maximum(bbox_tl[0], candidates_tl[:, 0]),
            torch.maximum(bbox_tl[1], candidates_tl[:, 1]),
        ],
        dim=1,
    )

    br = torch.stack(
        [
            torch.minimum(bbox_br[0], candidates_br[:, 0]),
            torch.minimum(bbox_br[1], candidates_br[:, 1]),
        ],
        dim=1,
    )
    wh = br - tl
    wh = torch.maximum(torch.zeros_like(wh), wh)
    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)

    iou_res = area_intersection / (
        area_bbox + area_candidates - area_intersection
    )
    return iou_res


def iou_cost(
    tracks: List[Track],
    detections: List[Detection],
    track_indices: Optional[List[int]] = None,
    detection_indices: Optional[List[int]] = None,
):
    """An intersection over union distance metric.

    Args:
        tracks : A list of tracks.
        detections : A list of detections.
        track_indices : A list of indices to tracks that should be matched.
            Defaults to all `tracks`.
        detection_indices : A list of indices to detections that should be
            matched. Defaults to all `detections`.

    Returns:
        cost_matrix: a cost matrix of shape
        [len(track_indices), len(detection_indices)], where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.
    """
    if track_indices is None:
        track_indices = torch.arange(len(tracks))
    if detection_indices is None:
        detection_indices = torch.arange(len(detections))

    cost_matrix = torch.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = torch.cat(
            [detections[i].tlwh.unsqueeze(0) for i in detection_indices], dim=0
        )
        iou_res = iou(bbox, candidates)
        cost_matrix[row, :] = torch.ones_like(iou_res) - iou_res
    return cost_matrix
