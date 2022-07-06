"""
Track graph of deep SORT.
Taken and modified from
https://github.com/nwojke/deep_sort/blob/master/deep_sort/track.py
"""
import copy
import sys
from collections import defaultdict
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, TypedDict, Union

import torch
from pytorch_lightning.utilities import rank_zero_warn
from scipy.optimize import linear_sum_assignment

from vis4d.common.bbox.utils import bbox_iou
from vis4d.model.track.graph import BaseTrackGraph
from vis4d.struct import Boxes2D, InputSample, LabelInstances, LossesType

from ..motion.kalman_filter import KalmanFilterModel, KalmanFilterModule


class TrackState(Enum):
    """
    Enumeration type for the single target track state. Newly created tracks
    are classified as `tentative` until enough evidence has been collected.
    Then, the track state is changed to `confirmed`. Tracks that are no longer
    alive are classified as `deleted` to mark them for removal from the set of
    active tracks.
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track(TypedDict):
    """Tracklet implementation."""

    motion_model: KalmanFilterModel
    #  A unique track identifier.
    track_id: int
    #  Number of consecutive detections before the track is confirmed.
    #  The track state is set to `Deleted` if a miss occurs within the first
    #  `n_init` frames.
    n_init: int
    # Max age of this track
    max_age: int
    #  A cache of features. On each measurement update, the associated feature
    #  vector is added to this list.
    features: List[torch.Tensor]
    # The current track state.
    state: TrackState
    # confidence of this detection
    confidence: float
    # class_id that belongs to this track
    class_id: int


class KalmanParams(TypedDict):
    """Parameters used to initialize kalman filter."""

    cov_motion_Q: List[float]
    cov_project_R: List[float]
    cov_P0: List[float]


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


class TrackManager:
    """Static class that provides utility function to interact with different
    track objects
    (converting bounding boxes, mark track as missed, etc.)"""

    @staticmethod
    def create_track(
        track_id: int,
        motion_model: KalmanFilterModel,
        n_init: int,
        max_age: int,
        confidence: float,
        class_id: int,
        feature: Optional[torch.Tensor] = None,
    ) -> Track:
        """Creates a new track object"""

        features = []
        if feature is not None:
            features.append(feature)

        track: Track = {
            "motion_model": motion_model,
            "track_id": track_id,
            "n_init": n_init,
            "max_age": max_age,
            "class_id": class_id,
            "features": features,
            "confidence": confidence,
            "state": TrackState.Tentative,
        }
        return track

    @staticmethod
    def mark_missed(track: Track) -> None:
        """Mark this track as missed (no association at the current time
        step)."""
        if (
            track["state"] == TrackState.Tentative
        ):  # This track has not been confirmed -> remove it directly
            track["state"] = TrackState.Deleted
        elif (
            track["motion_model"].time_since_update > track["max_age"]
        ):  # This track has not been observed for a long time -> delete
            track["state"] = TrackState.Deleted

    @staticmethod
    def get_track_as_tlbr_bbox(track: Track) -> List[float]:
        """Convert a single one dimension tlbr box to xyah.
        Returns:
            bbox_tlbr: shape (4,)
                        [top_left_x, top_left_y, bottom_right_x,
                        bottom_right_y]
        """
        bbox_tlbr = track["motion_model"].get_state()[:4].clone().detach()
        height = track["motion_model"].get_state()[3]
        half_width = (height * track["motion_model"].get_state()[2]) / 2.0
        half_height = height / 2.0

        bbox_tlbr[0] = track["motion_model"].get_state()[0] - half_width
        bbox_tlbr[1] = track["motion_model"].get_state()[1] - half_height
        bbox_tlbr[2] = track["motion_model"].get_state()[0] + half_width
        bbox_tlbr[3] = track["motion_model"].get_state()[1] + half_height
        return bbox_tlbr

    @staticmethod
    def tlbr_to_xyah(bbox_tlbr: torch.Tensor) -> torch.Tensor:
        """Convert a single one dimension tlbr box to xyah.
        Args:
            bbox_tlbr: shape (4,)
                        [top_left_x, top_left_y, bottom_right_x,
                        bottom_right_y]
        Returns:
            bbox_xyah: (4, ) [center_x, center_y, aspect_ratio, height]
        """
        bbox_xyah = bbox_tlbr.clone().detach()
        if len(bbox_tlbr.shape) > 1:  # batch mode
            bbox_xyah[:, :2] = (bbox_tlbr[:, 2:] + bbox_tlbr[:, :2]) / 2.0
            height = bbox_tlbr[:, 3] - bbox_tlbr[:, 1]
            width = bbox_tlbr[:, 2] - bbox_tlbr[:, 0]
            bbox_xyah[:, 2] = width / height
            bbox_xyah[:, 3] = height
        else:
            bbox_xyah[:2] = (bbox_tlbr[2:] + bbox_tlbr[:2]) / 2.0
            height = bbox_tlbr[3] - bbox_tlbr[1]
            width = bbox_tlbr[2] - bbox_tlbr[0]
            bbox_xyah[2] = width / height
            bbox_xyah[3] = height
        return bbox_xyah

    @staticmethod
    def xyah_to_tlbr(bbox_xyah: torch.Tensor) -> torch.Tensor:
        """Convert a single one dimension tlbr box to xyah.
        Args:
            bbox_xyah: (4, ) [center_x, center_y, aspect_ratio, height]
        Returns:
            bbox_tlbr: shape (4,)
                        [top_left_x, top_left_y, bottom_right_x,
                        bottom_right_y]
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
    """

    def __init__(
        self,
        matching_threshold: float,
        budget: Optional[int] = None,
    ):
        """Init."""
        self._metric = NearestNeighborDistanceMetric._nn_cosine_distance
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

    @staticmethod
    def _cosine_distance(
        matrix_a: List[torch.Tensor],
        matrix_b: List[torch.Tensor],
        data_is_normalized: bool = False,
    ) -> torch.Tensor:
        """Compute pair-wise cosine distance between points in `a` and `b`.
        Args:
            matrix_a : An NxL matrix of N samples of dimensionality L.
            matrix_b :  An MxL matrix of M samples of dimensionality L.
            data_is_normalized : If True, assumes rows in a and b are unit
        Returns:
            Returns a matrix of size NxM such that element (i, j)
            contains the squared distance between `a[i]` and `b[j]`.
        """

        samples_a = torch.stack(matrix_a, dim=0)
        samples_b = torch.stack(matrix_b, dim=0)
        if not data_is_normalized:
            samples_a = samples_a / torch.linalg.norm(
                samples_a, dim=1, keepdims=True
            )
            samples_b = samples_b / torch.linalg.norm(
                samples_b, dim=1, keepdims=True
            )
        return 1.0 - torch.matmul(samples_a, samples_b.T)

    @staticmethod
    def _nn_cosine_distance(
        matrix_x: List[torch.Tensor], matrix_y: List[torch.Tensor]
    ) -> torch.Tensor:
        """Helper function for nearest neighbor distance metric (cosine).
        Args:
            matrix_x : A matrix of N row-vectors (sample points).
            matrix_y : A matrix of M row-vectors (query points).
        Returns:
            min_distances: A vector of length M that contains for each entry
            in `y`
            the smallest cosine distance to a sample in `x`.
        """
        distances = NearestNeighborDistanceMetric._cosine_distance(
            matrix_x, matrix_y
        )
        min_distance = torch.min(distances, dim=0)[0]
        return min_distance


class CostMatching:
    """Static class that stores util functions for the cost matching"""

    @staticmethod
    def min_cost_matching(
        cost_matrix: torch.Tensor,
        max_distance: float,
        tracks: List[Track],
        detections: Boxes2D,
        track_ids: Optional[List[int]] = None,
        detection_indices: Optional[List[int]] = None,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Solve linear assignment problem.
        Args:
            cost_matrix : The distance metric function is given a list of
                tracks and detections as well as a list of N track indices
                and M detection indices.
                The metric should return the NxM dimensional
                cost matrix, where element (i, j) is the association cost
                between the i-th track in the given track indices and the j-th
                detection in the given detection_indices.
            max_distance : Gating threshold. Associations with cost larger than
                this value are disregarded.
            tracks :A list of predicted tracks at the current time step.
            detections : A list of detections at the current time step.
            track_ids :List of track ids that maps rows in `cost_matrix` to
            tracks.
            `tracks` (see description above).
            detection_indices : List of detection indices that maps columns in
            `cost_matrix` to detections in `detections` (see description
            above).
        Returns:
            matches: A list of matched track and detection indices.
            unmatched_tracks: A list of unmatched track indices.
            unmatched_detections: A list of unmatched detection indices.
        """
        if track_ids is None:
            track_ids = torch.tensor([t["track_id"] for t in tracks]).to(
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

    @staticmethod
    def matching_cascade(
        distance_metric: Callable[
            [
                List[Track],
                Boxes2D,
                torch.Tensor,
                List[int],
                List[int],
            ],
            torch.Tensor,
        ],
        max_distance: float,
        cascade_depth: int,
        tracks: List[Track],
        detections: Boxes2D,
        det_features: torch.Tensor,
        track_ids: Optional[List[int]] = None,
        detection_indices: Optional[List[int]] = None,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Run matching cascade.
        Args:
        distance_metric :
            The distance metric is given a list of tracks and detections as
            well as
            a list of N track indices and M detection indices. The metric
            should
            return the NxM dimensional cost matrix, where element (i, j) is the
            association cost between the i-th track in the given track
            indices and
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
            track_ids = [
                track["track_id"] for track in tracks
            ]  # pragma: no cover
        if detection_indices is None:
            detection_indices = list(
                range(len(detections))
            )  # pragma: no cover

        unmatched_detections = detection_indices
        matches = []
        for level in range(cascade_depth):
            if len(unmatched_detections) == 0:  # No detections left
                break

            track_ids_l = [
                k
                for k in track_ids
                if tracks[k]["motion_model"].time_since_update == 1 + level
            ]
            if len(track_ids_l) == 0:  # Nothing to match at this level
                continue
            cost_matrix = distance_metric(
                tracks,
                detections,
                det_features,
                track_ids_l,
                unmatched_detections,
            )
            (
                matches_l,
                _,
                unmatched_detections,
            ) = CostMatching.min_cost_matching(
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

    @staticmethod
    def gate_cost_matrix(
        kf: KalmanFilterModule,
        cost_matrix: torch.Tensor,
        tracks: List[Track],
        detections: Boxes2D,
        track_ids: List[int],
        detection_indices: List[int],
        gated_cost: Optional[float] = sys.maxsize,
        only_position: Optional[bool] = False,
    ) -> torch.Tensor:
        """Apply gate for matching.
        Invalidate infeasible entries in cost matrix based on the state
        distributions obtained by Kalman filtering.
        Args:
            kf: The Kalman filter.
            cost_matrix :
                The NxM dimensional cost matrix, where N is the number of track
                indices and M is the number of detection indices, such that
                entry
                 (i, j) is the association cost between `tracks[track_ids[
                 i]]` and
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
                Entries in the cost matrix corresponding to infeasible
                associations
                 are set this value. Defaults to a very large value.
            only_position :
                If True, only the x, y position of the state distribution is
                considered during gating. Defaults to False.
        Returns:
            cost_matrix: the modified cost matrix.
        """
        gating_dim = 2 if only_position else 4
        gating_threshold = chi2inv95[gating_dim]
        measurements = TrackManager.tlbr_to_xyah(
            detections.boxes[detection_indices][:, :4]
        )

        for row, track_id in enumerate(track_ids):
            track = tracks[track_id]
            gating_distance = kf.gating_distance(
                track["motion_model"].get_state(),
                track["motion_model"].get_covariance(),
                measurements,
                only_position,
            )
            cost_matrix[row, gating_distance > gating_threshold] = gated_cost
        return cost_matrix


# ============================================================================
# ============= Start Tracking Graph Implementation ==========================
# ============================================================================


class DeepSORTTrackGraph(BaseTrackGraph):
    """Create the deepsort tracking graph.
    Parameters
    ----------
    num_classes: How many semantic classes occur in the dataset
    kalman_filter_params: TypedStruct or Dict with TypedStruct containing
    initial parameters for the kalman filter.
                          If a dict is passed, there should be a key for each
                          semantic class
    min_confidence: Min confidence threshold for a detected bbox
    max_cosine_distance: Max Cosine distance
    max_age: Max age for a track in the graph.
        If there has not been a detection for a track for longer than max_age
        update steps, it will be removed
    n_init: Number of consecutive detections before the track is confirmed. The
            track state is set to `Deleted` if a miss occurs within the first
            `n_init` frames.
    nn_budget: Budget for nearest neighbour matching. Number of samples for
    each class to store in memory
    max_iou_distance: Max iou distance for the bbox matching step
    remove_deleted_tracks_interval: How often deleted tracks should be removed
    from memory

    """

    def __init__(
        self,
        num_classes: int,
        kalman_filter_params: Union[KalmanParams, Dict[int, KalmanParams]],
        min_confidence: float = 0.3,
        max_cosine_distance: float = 0.2,
        max_age: int = 5,
        n_init: int = 1,
        nn_budget: Optional[int] = 100,
        max_iou_distance: float = 0.7,
        remove_deleted_tracks_interval=1,
    ) -> None:
        """Init"""
        super().__init__()

        self.num_classes = num_classes
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.metric = NearestNeighborDistanceMetric(
            self.max_cosine_distance, self.nn_budget
        )
        self.min_confidence = min_confidence

        if not isinstance(kalman_filter_params, Dict):
            # Same cov matrix for all classes. Construct Dict with repeated
            # argument
            self.kalman_filter_params: Dict[int, KalmanParams] = {}
            for class_id in range(self.num_classes):
                self.kalman_filter_params[class_id] = kalman_filter_params
        else:
            self.kalman_filter_params: Dict[
                int, KalmanParams
            ] = kalman_filter_params

        # Initialize Kalman filter
        self.class_idx_to_kf = self._init_kalman()
        # ---- Track Management ----
        self.num_tracks = 0  # How many tracks are currently in memory
        self.tracks: List[Track] = []
        self.remove_deleted_tracks_interval = remove_deleted_tracks_interval
        self.maintenance_counter = 0

        self.reset()

    def _init_kalman(self) -> torch.nn.ModuleList:
        """
        Initializes class specific kalman filters. Creates a kalman filter
        for each semantic class
        """
        # Motion Matrix. (Discrete Motion Model, integrate velocity to
        # position)
        motion_matrix = torch.eye(8, 8)
        for i in range(4):
            motion_matrix[i, 4 + i] = 1.0

        measurement_matrix = torch.eye(
            4, 8
        )  # What is measured from the state (first 4 entries -> box)
        class_idx_to_kf = torch.nn.ModuleList()
        # Create one kalman filter (different noise matrices) for each class
        # (e.g. human, car, ...)
        for class_id in range(self.num_classes):
            parameters = self.kalman_filter_params.get(class_id, None)
            assert (
                parameters is not None
            ), "Missing kalman parameter for class: " + str(class_id)

            class_idx_to_kf.append(
                KalmanFilterModule(
                    motion_matrix,
                    measurement_matrix,
                    torch.diag(
                        torch.tensor((parameters["cov_motion_Q"]))
                    ).float(),
                    torch.diag(
                        torch.tensor((parameters["cov_project_R"]))
                    ).float(),
                    torch.diag(torch.tensor((parameters["cov_P0"]))).float(),
                )
            )
        return class_idx_to_kf

    def reset(self) -> None:
        """Reset tracks. Removes all tracks from memory"""
        self.num_tracks = 0
        self.tracks = []

    def get_tracks(self) -> Boxes2D:
        """Get active tracks at current frame."""
        track_boxes: List[torch.Tensor] = []
        class_ids: List[int] = []
        track_ids: List[int] = []

        for track in self.tracks:
            if (
                track["motion_model"].time_since_update >= self.max_age
                or track["state"] == TrackState.Deleted
            ):
                continue

            track_boxes.append(
                torch.tensor(
                    [
                        *TrackManager.get_track_as_tlbr_bbox(track),
                        track["confidence"],
                    ]
                ).unsqueeze(0)
            )
            class_ids.append(track["class_id"])
            track_ids.append(track["track_id"])

        track_boxes_as_tensor = (
            torch.cat(track_boxes)
            if len(track_boxes) > 0
            else torch.empty((0, 5))
        )
        class_ids_as_tensor = (
            torch.tensor(class_ids) if len(class_ids) > 0 else torch.empty(0)
        )
        track_ids_as_tensor = (
            torch.tensor(track_ids) if len(track_ids) > 0 else torch.empty(0)
        )

        return Boxes2D(
            track_boxes_as_tensor, class_ids_as_tensor, track_ids_as_tensor
        )

    def forward_train(
        self,
        inputs: List[InputSample],
        predictions: List[LabelInstances],
        targets: Optional[List[LabelInstances]],
        **kwargs: List[torch.Tensor],
    ) -> LossesType:
        """Forward of DeepSORTTrackGraph in training stage."""
        raise NotImplementedError  # This track graph has no parameters that
        # need to be learned

    def forward_test(
        self,
        inputs: InputSample,
        predictions: LabelInstances,
        embeddings: Optional[torch.Tensor] = None,
        **kwargs: torch.Tensor,
    ) -> LabelInstances:
        """Process inputs, match detections with existing tracks."""
        if len(predictions.boxes2d) > 1:
            rank_zero_warn(
                f"DeepSort Tracking Graph: Expected List of Boxes2D to have "
                f"dimension 1 got {len(predictions.boxes2d)}."
                f" Will only use the first entry!"
            )

        detections = predictions.boxes2d[0].clone()

        frame_index = inputs.metadata[0].frameIndex
        if frame_index is not None and frame_index == 0:
            self.reset()

        # Propagate all tracks for one timestep
        self.predict()

        # only select boxes with high enough confidence
        detections_selected = detections[
            detections.score and detections.score >= self.min_confidence
        ]

        if embeddings is not None and detections.score is not None:
            embeddings = embeddings.clone()
            self.update(
                detections_selected,
                embeddings[detections.score >= self.min_confidence],
            )
        else:
            self.update(detections_selected)

        # Return all boxes
        output = self.get_tracks()
        result = copy.deepcopy(predictions)
        result.boxes2d[0] = output
        return result

    def predict(self) -> None:
        """Propagate all tracks one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track["motion_model"].predict()

    def create_track(
        self,
        detection_box: torch.Tensor,
        class_id: int,
        confidence: float,
        features: Optional[torch.Tensor] = None,
    ):
        """
        Creates a new track for a given detection_box, class_id, confidence
        score and extraced features
        """
        kalman_model = KalmanFilterModel(
            self.class_idx_to_kf[class_id],
            detection_box,
            num_frames=-1,
            motion_dims=3,
        )
        self.tracks.append(
            TrackManager.create_track(
                track_id=self.num_tracks + 1,
                motion_model=kalman_model,
                n_init=self.n_init,
                max_age=self.max_age,
                confidence=confidence,
                class_id=class_id,
                feature=features,
            )
        )
        self.num_tracks += 1

    def clear_deleted_tracks(self):
        """Cleans up local memory. Removes all tracks that are in the
        'Deleted' state."""
        # IDs to keep
        new_track_list = [
            track
            for track in self.tracks
            if track["state"] != TrackState.Deleted
        ]
        # Re-enumerate ids
        for idx, track in enumerate(new_track_list):
            track["track_id"] = idx

        # overwrite tracks
        self.tracks = new_track_list
        self.num_tracks = len(self.tracks)

    def update(
        self, detections: Boxes2D, det_features: torch.tensor = None
    ) -> None:
        """Perform association and track management."""

        cls_detidx_mapping = defaultdict(list)
        for i, class_id in enumerate(detections.class_ids):
            cls_detidx_mapping[int(class_id)].append(i)

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = [], [], []

        # for i, class_id in enumerate(detections.class_ids):
        for class_id, detidx in cls_detidx_mapping.items():
            (
                matches_cls,
                unmatched_tracks_cls,
                unmatched_detections_cls,
            ) = self._match(detections, detidx, class_id, det_features)
            matches.extend(matches_cls)
            unmatched_tracks.extend(unmatched_tracks_cls)
            unmatched_detections.extend(unmatched_detections_cls)

        matched_det_set = set()
        unmatched_det_set = set()

        # These detections had an associated match. Update track information
        for track_id, det_idx in matches:
            assert det_idx not in matched_det_set
            matched_det_set.add(det_idx)

            track = self.tracks[track_id]
            # kf measurement update step and update the feature cache.
            new_pos = TrackManager.tlbr_to_xyah(detections.boxes[det_idx][:4])
            track["motion_model"].update(new_pos)
            # We simply overwrite the confidence score with the newest
            # confidence
            track["confidence"] = float(detections.boxes[det_idx][4])
            if det_features is not None:
                track["features"].append(det_features[det_idx])

            if (
                track["state"] == TrackState.Tentative
                and track["motion_model"].hits >= track["n_init"]
            ):
                track["state"] = TrackState.Confirmed

        # mark unmatched tracks to 'delete' in certain cases
        for track_id in unmatched_tracks:
            TrackManager.mark_missed(self.tracks[track_id])

        # These detections had no associated match. --> Create new tracks
        for det_idx in unmatched_detections:
            # Create new tracks
            det_box = TrackManager.tlbr_to_xyah(detections.boxes[det_idx][:4])
            confidence = float(detections.boxes[det_idx][4])
            class_id = int(detections.class_ids[det_idx])
            if det_features is not None:
                self.create_track(
                    det_box, class_id, confidence, det_features[det_idx]
                )
            else:
                self.create_track(det_box, class_id, confidence)

        # Remove tracks that were marked to remove
        self.maintenance_counter += 1
        if self.maintenance_counter >= self.remove_deleted_tracks_interval:
            self.clear_deleted_tracks()
            self.maintenance_counter = 0

        for unmatched_det in unmatched_detections:
            assert unmatched_det not in unmatched_det_set
            unmatched_det_set.add(unmatched_det)

        assert len(matched_det_set & unmatched_det_set) == 0
        assert len(matched_det_set | unmatched_det_set) == len(detections)

        if det_features is not None:
            # Update distance metric.
            active_targets = [
                track["track_id"]
                for track in self.tracks
                if track["state"] == TrackState.Confirmed
            ]
            features, targets = [], []
            for track in self.tracks:
                if not track["state"] == TrackState.Confirmed:
                    continue
                features += track["features"]
                targets += [track["track_id"]]
                track["features"] = []

            self.metric.partial_fit(
                features,
                targets,
                active_targets,
            )

    def _match(
        self,
        detections: Boxes2D,
        detection_indices: List[int],
        class_id: int,
        det_features: torch.tensor = None,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Matching.
        Returns  matches, unmatched_tracks, unmatched_detections"""
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            t["track_id"]
            for t in self.tracks
            if t["state"] == TrackState.Confirmed and t["class_id"] == class_id
        ]
        unconfirmed_tracks = [
            t["track_id"]
            for t in self.tracks
            if t["state"] != TrackState.Confirmed and t["class_id"] == class_id
        ]

        if det_features is not None:

            def gated_metric(
                tracks: List[Track],
                dets: Boxes2D,
                dets_features: torch.tensor,
                track_ids: List[int],
                detection_indices: List[int],
            ) -> torch.tensor:
                """Calculate cost matrix."""
                features = [dets_features[i] for i in detection_indices]
                # calculate cost matrix using deep feature
                cost_matrix = self.metric.distance(features, track_ids)
                # use mahalanobis distance to gate cost matrix
                cost_matrix = CostMatching.gate_cost_matrix(
                    self.class_idx_to_kf[class_id],
                    cost_matrix,
                    tracks,
                    dets,
                    track_ids,
                    detection_indices,
                )
                return cost_matrix

            # Associate confirmed tracks using appearance features.
            (
                matches_a,
                unmatched_tracks_a,
                unmatched_detections,
            ) = CostMatching.matching_cascade(
                gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.tracks,
                detections,
                det_features,
                confirmed_tracks,
                detection_indices,
            )

            # Associate remaining tracks with IOU
            # This helps to account for sudden appearance changes
            iou_track_candidates = unconfirmed_tracks + [
                k
                for k in unmatched_tracks_a
                if self.tracks[k]["motion_model"].time_since_update == 1
            ]
            unmatched_tracks_a = [
                k
                for k in unmatched_tracks_a
                if self.tracks[k]["motion_model"].time_since_update != 1
            ]

            tracks_as_bbox = torch.empty((0, 5)).to(detections.device)

            for track_id in iou_track_candidates:
                bbox_t = TrackManager.xyah_to_tlbr(
                    self.tracks[track_id]["motion_model"].get_state()[:4]
                )
                conf = self.tracks[track_id]["confidence"]
                bbox_t = torch.cat(
                    (bbox_t, torch.tensor([conf]).to(bbox_t.device))
                ).unsqueeze(0)
                tracks_as_bbox = torch.cat((tracks_as_bbox, bbox_t), dim=0)

            iou_track_candidates_box2d = Boxes2D(tracks_as_bbox)

            unmatch_detections_box2d = detections[unmatched_detections]
            iou_res = bbox_iou(
                iou_track_candidates_box2d, unmatch_detections_box2d
            )
            iou_cost_matrix = torch.ones_like(iou_res) - iou_res

            (
                matches_b,
                unmatched_tracks_b,
                unmatched_detections,
            ) = CostMatching.min_cost_matching(
                iou_cost_matrix,
                self.max_iou_distance,
                self.tracks,
                detections,
                iou_track_candidates,
                unmatched_detections,
            )

            matches = matches_a + matches_b
            unmatched_tracks = list(
                set(unmatched_tracks_a + unmatched_tracks_b)
            )

        else:
            bbox = torch.empty((0, 5)).to(detections.device)
            for _, track_id in enumerate(confirmed_tracks):
                bbox_t = TrackManager.xyah_to_tlbr(
                    self.tracks[track_id]["motion_model"].get_state()[:4]
                )
                conf = self.tracks[track_id]["confidence"]
                bbox_t = torch.cat(
                    (bbox_t, torch.tensor([conf]).to(bbox_t.device))
                ).unsqueeze(0)
                bbox = torch.cat((bbox, bbox_t), dim=0)
            track_box2d = Boxes2D(bbox)
            detections_box2d = detections[detection_indices]
            iou_res = bbox_iou(track_box2d, detections_box2d)
            iou_cost_matrix = torch.ones_like(iou_res) - iou_res

            (
                matches,
                unmatched_tracks,
                unmatched_detections,
            ) = CostMatching.min_cost_matching(
                iou_cost_matrix,
                self.max_iou_distance,
                self.tracks,
                detections,
                confirmed_tracks,
                detection_indices,
            )

        return matches, unmatched_tracks, unmatched_detections
