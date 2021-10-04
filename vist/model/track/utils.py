"""Tracking model utils."""
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment as linear_assignment

from vist.struct import InputSample
from vist.struct.labels import Boxes2D


def split_key_ref_inputs(
    batched_input_samples: List[List[InputSample]],
) -> Tuple[List[InputSample], List[List[InputSample]]]:
    """Split key / ref frame inputs from batched List of InputSample."""
    key_indices = []  # type: List[int]
    ref_indices = []  # type: List[List[int]]
    for input_samples in batched_input_samples:
        curr_ref_indices = list(range(0, len(input_samples)))
        for i, sample in enumerate(input_samples):
            if (
                sample.metadata.attributes is not None
                and sample.metadata.attributes.get("keyframe", False)
            ):
                key_indices.append(curr_ref_indices.pop(i))
                ref_indices.append(curr_ref_indices)
                break

    key_inputs = [
        inputs[key_index]
        for inputs, key_index in zip(batched_input_samples, key_indices)
    ]
    ref_inputs = [
        [inputs[i] for i in curr_ref_indices]
        for inputs, curr_ref_indices in zip(batched_input_samples, ref_indices)
    ]
    return key_inputs, ref_inputs


def cosine_similarity(
    key_embeds: torch.Tensor,
    ref_embeds: torch.Tensor,
    normalize: bool = True,
    temperature: float = -1,
) -> torch.Tensor:
    """Calculate cosine similarity."""
    if normalize:
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)

    dists = torch.mm(key_embeds, ref_embeds.t())

    if temperature > 0:
        dists /= temperature  # pragma: no cover
    return dists


class KalmanFilter(nn.Module):  # type: ignore
    """Kalman filter.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    """

    def __init__(
        self,
        motion_mat: torch.Tensor,
        update_mat: torch.Tensor,
        cov_motion_Q: torch.Tensor,
        cov_project_R: torch.Tensor,
        cov_P0: torch.Tensor,
    ) -> None:
        """Init."""
        super().__init__()
        self.register_buffer("_motion_mat", motion_mat)
        self.register_buffer("_update_mat", update_mat)
        self.register_buffer("_cov_motion_Q", cov_motion_Q)
        self.register_buffer("_cov_project_R", cov_project_R)
        self.register_buffer("_cov_P0", cov_P0)

    def initiate(
        self,
        measurement: torch.Tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Initiate a Kalman filter state based on the first measurement

        Args:
            measurement: Bounding box coordinates (x, y, a, h) with center
                position (x, y), aspect ratio a, and height h.

        Returns:
            mean, covariance: the mean vector (8 dimensional) and covariance
                matrix (8x8 dimensional) of the new track. Unobserved
                velocities are initialized to 0 mean.
        """
        mean_pos = measurement.clone().detach()
        mean_vel = torch.zeros_like(mean_pos)
        mean = torch.cat([mean_pos, mean_vel], dim=0)

        covariance = self._cov_P0
        return mean, covariance

    def predict(
        self,
        mean: torch.tensor,
        covariance: torch.tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Run Kalman filter prediction step.

        Args:
            mean: The 8 dimensional mean vector of the object state at the
                previous time step.
            covariance: The 8x8 dimensional covariance matrix of the object
                state at the previous time step.

        Returns:
            mean: the mean vector, Unobserved velocities are initialized to
                0 mean.
            covariance: covariance matrix of the predicted state.
        """
        mean = torch.matmul(self._motion_mat, mean)
        covariance = (
            torch.matmul(
                self._motion_mat, torch.matmul(covariance, self._motion_mat.T)
            )
            + self._cov_motion_Q
        )

        return mean, covariance

    def project(
        self, mean: torch.tensor, covariance: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Project state distribution to measurement space.

        Args:
            mean :
                The state's mean vector (8 dimensional vector).
            covariance :
                The state's covariance matrix (8x8 dimensional).

        Returns:
            mean: the projected mean of the given state estimate.
            projected_cov: the projected covariance matrix of the given state
                estimate.
        """
        mean = torch.matmul(self._update_mat, mean)
        covariance = torch.matmul(
            self._update_mat, torch.matmul(covariance, self._update_mat.T)
        )
        projected_cov = covariance + self._cov_project_R
        return mean, projected_cov

    def update(
        self,
        mean: torch.tensor,
        covariance: torch.tensor,
        measurement: torch.tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Run Kalman filter correction step.

        Args:
            mean :
                The predicted state's mean vector (8 dimensional).
            covariance :
                The state's covariance matrix (8x8 dimensional).
            measurement :
                The 4 dimensional measurement vector (x, y, a, h), where (x, y)
                is the center position, a the aspect ratio, and h the height of
                the bounding box.

        Returns:
            new_mean: updated mean
            new_covariance: updated covariance
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor = torch.cholesky(projected_cov)
        kalman_gain = torch.cholesky_solve(
            torch.matmul(covariance, self._update_mat.T).T,
            chol_factor,
            upper=False,
        ).T

        innovation = measurement - projected_mean

        new_mean = mean + torch.matmul(innovation, kalman_gain.T)
        new_covariance = covariance - torch.matmul(
            kalman_gain, torch.matmul(projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: torch.tensor,
        covariance: torch.tensor,
        measurements: torch.tensor,
        only_position: Optional[bool] = False,
    ) -> torch.tensor:
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Args:
            mean :
                Mean vector over the state distribution (8 dimensional).
            covariance :
                Covariance of the state distribution (8x8 dimensional).
            measurements :
                An Nx4 dimensional matrix of N measurements, each in
                format (x, y, a, h) where (x, y) is the bounding box center
                position, a the aspect ratio, and h the height.
            only_position: If True, distance computation is done with respect
                to the bounding box center position only.

        Returns:
            squared_maha: a vector of length N, where the i-th element contains
                the squared Mahalanobis distance between (mean, covariance) and
                `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        cholesky_factor = torch.cholesky(covariance)
        d = measurements - mean
        z = torch.triangular_solve(
            d.T,
            cholesky_factor,
            upper=False,
        )[0]
        squared_maha = torch.sum(z * z, axis=0)
        return squared_maha


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
            only_position,
        )
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix


def _cosine_distance(
    a: List[torch.tensor],
    b: List[torch.tensor],
    data_is_normalized: bool = False,
) -> torch.tensor:  # pylint: disable = invalid-name
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Args:
        a : An NxL matrix of N samples of dimensionality L.
        b :  An MxL matrix of M samples of dimensionality L.
        data_is_normalized : If True, assumes rows in a and b are unit length
            vectors. Otherwise, a and b are explicitly normalized to lenght 1.

    Returns:
        Returns a matrix of size NxM such that element (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        samples_a = torch.stack(a, dim=0)
        normed_a = samples_a / torch.linalg.norm(
            samples_a, dim=1, keepdims=True
        )
        samples_b = torch.stack(b, dim=0)
        normed_b = samples_b / torch.linalg.norm(
            samples_b, dim=1, keepdims=True
        )
    return 1.0 - torch.matmul(normed_a, normed_b.T)


def _nn_cosine_distance(
    x: List[torch.tensor], y: List[torch.tensor]
) -> torch.tensor:  # pylint: disable = invalid-name
    """Helper function for nearest neighbor distance metric (cosine).

    Args:
        x : A matrix of N row-vectors (sample points).
        y : A matrix of M row-vectors (query points).

    Returns:
        min_distances: A vector of length M that contains for each entry in `y`
        the smallest cosine distance to a sample in `x`.
    """
    distances = _cosine_distance(x, y)
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
        self.samples: Dict[int, List[torch.tensor]] = {}

    def partial_fit(
        self,
        features: List[torch.tensor],
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
        self, features: List[torch.tensor], targets: List[int]
    ) -> torch.tensor:
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
