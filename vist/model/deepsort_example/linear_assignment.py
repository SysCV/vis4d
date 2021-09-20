"""LInear assignment."""
from __future__ import absolute_import

from typing import Callable, List, Optional, Tuple, Dict, Any
import torch
from scipy.optimize import linear_sum_assignment as linear_assignment

from vist.struct.labels import Boxes2D
from .kalman_filter import KalmanFilter, chi2inv95


INFTY_COST = 1e5


def tlbr_to_xyah(bbox_tlbr: torch.tensor) -> torch.tensor:
    """Convert a batched tlbr box to xyah.

    Args:
        bbox_tlbr: shape (N, 4)
                    [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

    Returns:
        bbox_xyah: (N, 4 ) [center_x, center_y, aspect_ratio, height]
    """
    bbox_xyah = bbox_tlbr.clone().detach()
    bbox_xyah[:, :2] = (bbox_tlbr[:, 2:] + bbox_tlbr[:, :2]) / 2.0
    height = bbox_tlbr[:, 3] - bbox_tlbr[:, 1]
    width = bbox_tlbr[:, 2] - bbox_tlbr[:, 0]
    bbox_xyah[:, 2] = width / height
    bbox_xyah[:, 3] = height
    return bbox_xyah


def min_cost_matching(
    cost_matrix,
    max_distance: float,
    tracks: Dict[int, Dict[str, Any]],
    detections: Boxes2D,
    track_ids: Optional[List[int]] = None,
    detection_indices: Optional[List[int]] = None,
) -> Tuple[
    List[Tuple[int, int]], List[int], List[int]
]:  # pylint: disable= line-too-long
    """Solve linear assignment problem.

    Args:
        distance_metric : The distance metric function is given a list of
            tracks and detections as well as a list of N track indices and M
            detection indices. The metric should return the NxM dimensional
            cost matrix, where element (i, j) is the association cost between
            the i-th track in the given track indices and the j-th detection
            in the given detection_indices.
        max_distance : Gating threshold. Associations with cost larger than
            this value are disregarded.
        tracks :A list of predicted tracks at the current time step.
        detections : A list of detections at the current time step.
        track_ids :List of track ids that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
        detection_indices : List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns:
        matches: A list of matched track and detection indices.
        unmatched_tracks: A list of unmatched track indices.
        unmatched_detections: A list of unmatched detection indices.
    """
    if track_ids is None:
        track_ids = torch.tensor(list(tracks.keys())).to(detections.device)
    if detection_indices is None:
        detection_indices = torch.arange(len(detections)).to(detections.device)

    if len(detection_indices) == 0 or len(track_ids) == 0:
        return [], track_ids, detection_indices  # Nothing to match.

    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    cost_matrix = cost_matrix.cpu()
    row_indices, col_indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_id in enumerate(track_ids):
        if row not in row_indices:
            unmatched_tracks.append(track_id)
    for row, col in zip(row_indices, col_indices):
        track_id = track_ids[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_id)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_id, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
    distance_metric: Callable[
        [
            Dict[int, Dict[str, Any]],
            Boxes2D,
            torch.tensor,
            List[int],
            List[int],
        ],
        torch.tensor,
    ],
    max_distance: float,
    cascade_depth: int,
    tracks: Dict[int, Dict[str, Any]],
    detections: Boxes2D,
    det_features: torch.tensor,
    track_ids: Optional[List[int]] = None,
    detection_indices: Optional[List[int]] = None,
) -> Tuple[
    List[Tuple[int, int]], List[int], List[int]
]:  # pylint: disable =line-too-long
    """Run matching cascade.

    Args:
    distance_metric :
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_ids : Optional[List[int]]
        List of track ids that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns:
        matches:A list of matched track and detection indices.
        unmatched_tracks: A list of unmatched track indices.
        unmatched_detections: A list of unmatched detection indices.
    """
    if track_ids is None:
        track_ids = list(tracks.keys())
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_ids_l = [
            k for k in track_ids if tracks[k]["time_since_update"] == 1 + level
        ]
        if len(track_ids_l) == 0:  # Nothing to match at this level
            continue

        cost_matrix = distance_metric(
            tracks, detections, det_features, track_ids_l, unmatched_detections
        )
        matches_l, _, unmatched_detections = min_cost_matching(
            cost_matrix,
            max_distance,
            tracks,
            detections,
            track_ids_l,
            unmatched_detections,
        )
        matches += matches_l
    unmatched_tracks = list(set(track_ids) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
    kf: KalmanFilter,
    cost_matrix: torch.tensor,
    tracks: Dict[int, Dict[str, Any]],
    detections: Boxes2D,
    track_ids: List[int],
    detection_indices: List[int],
    gated_cost: Optional[float] = INFTY_COST,
    only_position: Optional[bool] = False,
) -> torch.tensor:
    """Apply gate for matching.

    Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Args:
        kf: The Kalman filter.
        cost_matrix :
            The NxM dimensional cost matrix, where N is the number of track indices
            and M is the number of detection indices, such that entry (i, j) is the
            association cost between `tracks[track_ids[i]]` and
            `detections[detection_indices[j]]`.
        tracks :
            A list of predicted tracks at the current time step.
        detections : List[detection.Detection]
            A list of detections at the current time step.
        track_ids :
            List of track ids that maps rows in `cost_matrix` to tracks in
            `tracks` (see description above).
        detection_indices :
            List of detection indices that maps columns in `cost_matrix` to
            detections in `detections` (see description above).
        gated_cost :
            Entries in the cost matrix corresponding to infeasible associations are
            set this value. Defaults to a very large value.
        only_position :
            If True, only the x, y position of the state distribution is considered
            during gating. Defaults to False.

    Returns:
        cost_matrix: the modified cost matrix.
    """
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = tlbr_to_xyah(detections.boxes[detection_indices][:, :4])
    for row, track_id in enumerate(track_ids):
        track = tracks[track_id]
        gating_distance = kf.gating_distance(
            track["mean"],
            track["covariance"],
            measurements,
            track["class_id"],
            only_position,
        )
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix
