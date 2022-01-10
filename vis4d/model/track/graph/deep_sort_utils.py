"""DeepSORT utils."""
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from scalabel.label.io import load
from scipy.optimize import linear_sum_assignment  # type: ignore

from vis4d.model.track.motion import KalmanFilter
from vis4d.struct import Boxes2D


def tlbr_to_xyah(bbox_tlbr: torch.Tensor) -> torch.Tensor:
    """Convert a single one dimension tlbr box to xyah.

    Args:
        bbox_tlbr: shape (4,)
                    [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

    Returns:
        bbox_xyah: (4, ) [center_x, center_y, aspect_ratio, height]
    """
    bbox_xyah = bbox_tlbr.clone().detach()
    bbox_xyah[:2] = (bbox_tlbr[2:] + bbox_tlbr[:2]) / 2.0
    height = bbox_tlbr[3] - bbox_tlbr[1]
    width = bbox_tlbr[2] - bbox_tlbr[0]
    bbox_xyah[2] = width / height
    bbox_xyah[3] = height
    return bbox_xyah


def batch_tlbr_to_xyah(bbox_tlbr: torch.Tensor) -> torch.Tensor:
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


def xyah_to_tlbr(bbox_xyah: torch.Tensor) -> torch.Tensor:
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


INFTY_COST = 1e5

# Table for the 0.95 quantile of the chi-square distribution with N degrees of
# freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
# function and used as Mahalanobis gating threshold.

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


def min_cost_matching(
    cost_matrix: torch.Tensor,
    max_distance: float,
    tracks: Dict[int, Dict[str, Union[int, float, torch.Tensor]]],
    detections: Boxes2D,
    track_ids: Optional[List[int]] = None,
    detection_indices: Optional[List[int]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Solve linear assignment problem.

    Args:
        cost_matrix : The distance metric function is given a list of
            tracks and detections as well as a list of N track indices and M
            detection indices. The metric should return the NxM dimensional
            cost matrix, where element (i, j) is the association cost between
            the i-th track in the given track indices and the j-th detection
            in the given detection_indices.
        max_distance : Gating threshold. Associations with cost larger than
            this value are disregarded.
        tracks :A list of predicted tracks at the current time step.
        detections : A list of detections at the current time step.
        track_ids :List of track ids that maps rows in `cost_matrix` to tracks.
        `tracks` (see description above).
        detection_indices : List of detection indices that maps columns in
        `cost_matrix` to detections in `detections` (see description above).

    Returns:
        matches: A list of matched track and detection indices.
        unmatched_tracks: A list of unmatched track indices.
        unmatched_detections: A list of unmatched detection indices.
    """
    if track_ids is None:
        track_ids = torch.tensor(list(tracks.keys())).to(
            detections.device
        )  # pragma: no cover
    if detection_indices is None:
        detection_indices = torch.arange(len(detections)).to(
            detections.device
        )  # pragma: no cover
    if len(detection_indices) == 0 or len(track_ids) == 0:
        return [], track_ids, detection_indices  # Nothing to match.

    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    cost_matrix = cost_matrix.cpu()
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

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
            Dict[int, Dict[str, Union[int, float, torch.Tensor]]],
            Boxes2D,
            torch.Tensor,
            List[int],
            List[int],
        ],
        torch.Tensor,
    ],
    max_distance: float,
    cascade_depth: int,
    tracks: Dict[int, Dict[str, Union[int, float, torch.Tensor]]],
    detections: Boxes2D,
    det_features: torch.Tensor,
    track_ids: Optional[List[int]] = None,
    detection_indices: Optional[List[int]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
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
        track_ids = list(tracks.keys())  # pragma: no cover
    if detection_indices is None:
        detection_indices = list(range(len(detections)))  # pragma: no cover

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
    cost_matrix: torch.Tensor,
    tracks: Dict[int, Dict[str, Union[int, float, torch.Tensor]]],
    detections: Boxes2D,
    track_ids: List[int],
    detection_indices: List[int],
    gated_cost: Optional[float] = INFTY_COST,
    only_position: Optional[bool] = False,
) -> torch.Tensor:
    """Apply gate for matching.

    Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Args:
        kf: The Kalman filter.
        cost_matrix :
            The NxM dimensional cost matrix, where N is the number of track
            indices and M is the number of detection indices, such that entry
             (i, j) is the association cost between `tracks[track_ids[i]]` and
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
            Entries in the cost matrix corresponding to infeasible associations
             are set this value. Defaults to a very large value.
        only_position :
            If True, only the x, y position of the state distribution is
            considered during gating. Defaults to False.

    Returns:
        cost_matrix: the modified cost matrix.
    """
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = batch_tlbr_to_xyah(
        detections.boxes[detection_indices][:, :4]
    )
    for row, track_id in enumerate(track_ids):
        track = tracks[track_id]
        gating_distance = kf.gating_distance(
            track["mean"],
            track["covariance"],
            measurements,
            only_position,
        )
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix


def _cosine_distance(
    matrix_a: List[torch.Tensor],
    matrix_b: List[torch.Tensor],
    data_is_normalized: bool = False,
) -> torch.Tensor:
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Args:
        matrix_a : An NxL matrix of N samples of dimensionality L.
        matrix_b :  An MxL matrix of M samples of dimensionality L.
        data_is_normalized : If True, assumes rows in a and b are unit length
            vectors. Otherwise, a and b are explicitly normalized to lenght 1.

    Returns:
        Returns a matrix of size NxM such that element (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        samples_a = torch.stack(matrix_a, dim=0)
        normed_a = samples_a / torch.linalg.norm(
            samples_a, dim=1, keepdims=True
        )
        samples_b = torch.stack(matrix_b, dim=0)
        normed_b = samples_b / torch.linalg.norm(
            samples_b, dim=1, keepdims=True
        )
    return 1.0 - torch.matmul(normed_a, normed_b.T)


def _nn_cosine_distance(
    matrix_x: List[torch.Tensor], matrix_y: List[torch.Tensor]
) -> torch.Tensor:
    """Helper function for nearest neighbor distance metric (cosine).

    Args:
        matrix_x : A matrix of N row-vectors (sample points).
        matrix_y : A matrix of M row-vectors (query points).

    Returns:
        min_distances: A vector of length M that contains for each entry in `y`
        the smallest cosine distance to a sample in `x`.
    """
    distances = _cosine_distance(matrix_x, matrix_y)
    min_distance = torch.min(distances, dim=0)[0]
    return min_distance


class NearestNeighborDistanceMetric:
    """A nearest neighbor distance metric.

    For each target, returns the closest distance to any sample that has been
    observed so far.

    Args:
        matching_threshold: float
            The matching threshold. Samples with larger distance are considered
            an invalid match.
        budget : Optional[int]
            If not None, fix samples per class to at most this number. Removes
            the oldest samples when the budget is reached.
        samples : Dict[int -> List[ndarray]]
            A dictionary that maps from target identities to the list of
            samples that have been observed so far.
    """

    def __init__(
        self,
        matching_threshold: float,
        budget: Optional[int] = None,
    ):
        """Init."""
        self._metric = _nn_cosine_distance
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples: Dict[int, List[torch.Tensor]] = {}

    def partial_fit(
        self,
        features: List[torch.Tensor],
        targets: List[int],
        active_targets: List[int],
    ) -> None:
        """Update the distance metric with new data.

        Args:
            features: An NxM matrix of N features of dimensionality M.
            targets: An integer tensor of associated target identities.
            active_targets: A list of targets that are currently present in the
                scene.
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-1 * self.budget :]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(
        self, features: List[torch.Tensor], targets: List[int]
    ) -> torch.Tensor:
        """Compute distance between features and targets.

        Args:
            features :An NxL matrix of N features of dimensionality L.
            targets : A list of targets to match the given `features` against.

        Returns:
            cost_matrix: a cost matrix of shape [len(targets), len(features)],
            where element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.
        """
        cost_matrix = torch.empty((0, len(features))).to(features[0].device)
        for _, target in enumerate(targets):
            min_dist = self._metric(self.samples[target], features)
            cost_matrix = torch.cat(
                (cost_matrix, min_dist.unsqueeze(0)), dim=0
            )
        return cost_matrix


def load_predictions(
    pred_path: str, category_mapping: Dict[str, int]
) -> Dict[str, Boxes2D]:
    """Load scalabel format predictions into Vis4D."""
    preds_per_frame: Dict[str, Boxes2D] = {}
    given_predictions = load(pred_path).frames
    for prediction in given_predictions:
        name = prediction.name
        if prediction.labels is None:
            preds_per_frame[name] = Boxes2D(
                torch.empty((0, 5))
            )  # pragma: no cover
        else:
            preds_per_frame[name] = Boxes2D.from_scalabel(
                prediction.labels,
                category_mapping,
            )
    return preds_per_frame
