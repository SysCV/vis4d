"""Match based on IOU."""
from typing import List, Tuple

import torch
from torch import nn
from detectron2.structures import Boxes, pairwise_iou
from scipy.optimize import linear_sum_assignment as linear_assignment

balance_lambda: float = 0.0
max_IOU_distance: float = 0.7


def match(
    tracks_boxes: torch.Tensor,
    det_boxes: torch.Tensor,
    tracks_indices,
    det_indices,
    det_features,
    track_features,
) -> Tuple[
    List[Tuple[torch.LongTensor, torch.LongTensor]],
    List[torch.LongTensor],
    List[torch.LongTensor],
]:
    """Match detections and existing tracks based on bbox IOU.

    tracks_boxes: torch.Tensor(Nx4), xyxy
    det_boxes: torch.Tensor(Nx4), xyxy
    """
    iou_matrix = pairwise_iou(Boxes(tracks_boxes), Boxes(det_boxes))
    iou_cost_matrix = torch.ones_like(iou_matrix) - iou_matrix

    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    cos_similarity = cos(track_features.unsqueeze(1), det_features)
    feature_cost_matrix = torch.ones_like(cos_similarity) - cos_similarity

    cost_matrix = (
        balance_lambda * iou_cost_matrix
        + (1 - balance_lambda) * feature_cost_matrix
    )
    row_indices, col_indices = linear_assignment(cost_matrix)
    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(det_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(tracks_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = tracks_indices[row]
        detection_idx = det_indices[col]
        if iou_cost_matrix[row, col] > max_IOU_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections
