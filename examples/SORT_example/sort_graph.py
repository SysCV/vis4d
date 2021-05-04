"""Track graph of SORT."""
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
import scipy.linalg

from openmt.struct import Boxes2D
from openmt.model.track.graph import BaseTrackGraph, TrackGraphConfig
from openmt.common.bbox.utils import compute_iou


class SORTTrackGraphConfig(TrackGraphConfig):
    """SORT graph config."""

    keep_in_memory: int = 1
    init_score_thr: float = 0.7
    obj_score_thr: float = 0.3
    nms_backdrop_iou_thr: float = 0.3
    nms_class_iou_thr: float = 0.7


def boxes_to_xyah(boxes: torch.Tensor) -> torch.FloatTensor:
    """Convert boxes to xyah

    boxes: torch.FloatTensor: (N, 5) where each entry is defined by
    [x1, y1, x2, y2, score]
    """
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    x = 0.5 * (boxes[:, 2] + boxes[:, 0])
    y = 0.5 * (boxes[:, 3] + boxes[:, 1])
    scores = boxes[:, 4]
    xyah = torch.cat((x, y, width / height, height, scores), dim=1)
    return xyah


def xyah_to_boxes(xyah: torch.Tensor):
    height = xyah[:, 3]
    width = xyah[:, 2] * xyah[:, 3]
    x1 = xyah[:, 0] - width / 2
    x2 = xyah[:, 0] + width / 2
    y1 = xyah[:, 0] - height / 2
    y2 = xyah[:, 0] + height / 2
    scores = xyah[:, 4]
    boxes = torch.cat((x1, y1, x2, y2, scores), dim=1)
    return boxes


class SORTTrackGraph(BaseTrackGraph):
    """SORT tracking logic."""

    def __init__(self, cfg: TrackGraphConfig) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg = SORTTrackGraphConfig(**cfg.dict())
        self.kf = KalmanFilter()

    def get_tracks(
        self, frame_id: Optional[int] = None
    ) -> Tuple[Boxes2D, torch.Tensor]:
        """Get active tracks at given frame.

        If frame_id is None, return all tracks in memory.
        """
        bboxs, cls, ids, velocities, covariances = [], [], [], [], []
        for k, v in self.tracks.items():
            if frame_id is None or v["last_frame"] == frame_id:
                bboxs.append(v["bbox"].unsqueeze(0))
                cls.append(v["class_id"])
                ids.append(k)
                velocities.append(v["velocity"])
                covariances.append(v["covariance"])

        bboxs = torch.cat(bboxs) if len(bboxs) > 0 else torch.empty(0, 5)
        cls = torch.cat(cls) if len(cls) > 0 else torch.empty(0)
        ids = torch.tensor(ids).to(bboxs.device)  # type: ignore
        return Boxes2D(bboxs, cls, ids), velocities, covariances

    def forward(  # type: ignore # pylint: disable=arguments-differ
        self, detections: Boxes2D, frame_id: int
    ) -> Boxes2D:
        """Process inputs, match detections with existing tracks."""
        _, inds = detections.boxes[:, -1].sort(descending=True)
        detections = detections[inds, :].to(torch.device("cpu"))
        # duplicate removal for potential backdrops and cross classes
        valids = torch.full((len(detections),), 1)
        ious = compute_iou(detections, detections)
        for i in range(1, len(detections)):
            if detections.boxes[i, -1] < self.cfg.obj_score_thr:
                thr = self.cfg.nms_backdrop_iou_thr
            else:
                thr = self.cfg.nms_class_iou_thr

            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        detections = detections[valids, :]

        # init ids container
        ids = torch.full((len(detections),), -1, dtype=torch.long)
        # match if buffer is not empty
        det_bboxes = detections.boxes
        det_cls_ids = detections.class_ids
        det_cls_unique = torch.unique(det_cls_ids)

        memo_dets, velocities, covariances = self.get_tracks(frame_id - 1)
        memo_scores = memo_dets[:, -1]

        track_cls_ids = memo_dets.class_ids
        track_ids = memo_dets.track_ids
        track_cls_unique = torch.unique(track_cls_ids)

        kalman_states = torch.cat(
            (boxes_to_xyah(memo_dets.boxes)[:, :-1], velocities), dim=1
        )

        for existing_cls in track_cls_unique:
            km_state_per_cls = kalman_states[track_cls_ids == existing_cls]
            covariances_per_cls = covariances[track_cls_ids == existing_cls]
            det_per_cls = det_bboxes[det_cls_ids == existing_cls]

            pred_km_state_per_cls, pred_covariances_per_cls = kf.predict(
                km_state_per_cls, covariances_per_cls
            )

            self._match()
            self.mean, self.covariance = kf.update(
                det_per_cls, self.covariance, detections.boxes
            )

        new_inds = (ids == -1) & (
            detections.boxes[:, -1] > self.cfg.init_score_thr
        ).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracks, self.num_tracks + num_news, dtype=torch.long
        )
        self.num_tracks += num_news

        self.update(ids, detections, frame_id)
        result, _ = self.get_tracks(frame_id)
        return result

    def update(self, ids, detections, frame_id) -> None:  # type: ignore
        """Update track memory using matched detections."""
        tracklet_inds = ids > -1

        # update memo
        for cur_id, det in zip(  # type: ignore
            ids[tracklet_inds], detections[tracklet_inds]
        ):
            cur_id = int(cur_id)
            if cur_id in self.tracks.keys():
                self.update_track(cur_id, det, frame_id)
            else:
                self.create_track(cur_id, det, frame_id)

        # delete invalid tracks from memory
        invalid_ids = []
        for k, v in self.tracks.items():
            if frame_id - v["last_frame"] >= self.cfg.keep_in_memory:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    def update_track(
        self,
        track_id: int,
        detection: Boxes2D,
        frame_id: int,
        velocity: torch.Tensor,
        covariance: torch.Tensor,
    ) -> None:
        """Update a specific track with a new detection."""
        bbox, cls = detection.boxes[0], detection.class_ids[0]
        self.tracks[track_id]["bbox"] = bbox
        self.tracks[track_id]["last_frame"] = frame_id
        self.tracks[track_id]["class_id"] = cls
        self.tracks[track_id]["velocity"] = velocity
        self.tracks[track_id]["covariance"] = covariance

    def create_track(
        self,
        track_id: int,
        detection: Boxes2D,
        frame_id: int,
    ) -> None:
        """Create a new track from a detection."""
        bbox, cls = detection.boxes[0], detection.class_ids[0]

        _, covariance = self.kf.initiate(boxes_to_xyah(bbox))
        self.tracks[track_id] = dict(
            bbox=bbox,
            class_id=cls,
            last_frame=frame_id,
            velocity=torch.zeros_like(bbox),
            covariance=covariance,
        )


def iou_cost(tracks, detections, track_indices, detection_indices):
    def iou(bbox, candidates):
        bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
        candidates_tl = candidates[:, :2]
        candidates_br = candidates[:, :2] + candidates[:, 2:]

        tl = np.c_[
            np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
            np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis],
        ]
        br = np.c_[
            np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
            np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis],
        ]
        wh = np.maximum(0.0, br - tl)

        area_intersection = wh.prod(axis=1)
        area_bbox = bbox[2:].prod()
        area_candidates = candidates[:, 2:].prod(axis=1)
        return area_intersection / (
            area_bbox + area_candidates - area_intersection
        )

    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = 1e5
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray(
            [detections[i].tlwh for i in detection_indices]
        )
        cost_matrix[row, :] = 1.0 - iou(bbox, candidates)
    return cost_matrix


def min_cost_matching(
    distance_metric,
    max_distance,
    tracks,
    detections,
    track_indices=None,
    detection_indices=None,
):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices
    )
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    row_indices, col_indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


class KalmanFilter(object):
    """
    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot(
                (self._motion_mat, covariance, self._motion_mat.T)
            )
            + motion_cov
        )

        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance
