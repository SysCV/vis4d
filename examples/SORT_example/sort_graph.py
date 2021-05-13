"""Track graph of SORT."""
from typing import Dict, List, Optional, Tuple
import torch
from scipy.optimize import linear_sum_assignment as linear_assignment
import scipy.linalg

from openmt.struct import Boxes2D
from openmt.model.track.graph import BaseTrackGraph, TrackGraphConfig
from detectron2.structures import Boxes, pairwise_iou

# MetadataCatalog.get(dataset_name).idx_to_class_mapping


class SORTTrackGraphConfig(TrackGraphConfig):
    """SORT graph config."""

    max_IOU_distance: float = 0.7


def xyxy_to_xyah(xyxy: torch.Tensor) -> torch.FloatTensor:
    """Convert xyxy boxes to xya.

    xyxy: torch.FloatTensor: (N, 4) where each entry is defined by
    [x1, y1, x2, y2]
    """
    width = xyxy[:, [2]] - xyxy[:, [0]]
    height = xyxy[:, [3]] - xyxy[:, [1]]
    x = 0.5 * (xyxy[:, [2]] + xyxy[:, [0]])
    y = 0.5 * (xyxy[:, [3]] + xyxy[:, [1]])
    xyah = torch.cat((x, y, width / height, height), dim=1)
    return xyah


def xyah_to_xyxy(xyah: torch.Tensor):
    """Convert xyah boxes to xyxy.

    xyah: torch.FloatTensor: (4) where each entry is defined by
    [x1, y1, a, h]
    """
    height = xyah[:, [3]]
    width = xyah[:, [2]] * xyah[:, [3]]
    x1 = xyah[:, [0]] - width / 2
    x2 = xyah[:, [0]] + width / 2
    y1 = xyah[:, [1]] - height / 2
    y2 = xyah[:, [1]] + height / 2
    xyxy = torch.cat(
        (
            x1,
            y1,
            x2,
            y2,
        ),
        dim=1,
    )
    return xyxy


class SORTTrackGraph(BaseTrackGraph):
    """SORT tracking logic."""

    def __init__(self, cfg: TrackGraphConfig) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg = SORTTrackGraphConfig(**cfg.dict())
        self.kf = KalmanFilter()

    def get_tracks(
        self, frame_id: Optional[int] = None
    ) -> Tuple[Boxes2D, torch.Tensor, torch.Tensor]:
        """Get active tracks at given frame.

        If frame_id is None, return all tracks in memory.
        """
        bboxs, cls, ids, velocities, covariances = [], [], [], [], []
        for k, v in self.tracks.items():
            if frame_id is None or v["last_frame"] == frame_id:
                bboxs.append(v["bbox"].unsqueeze(0))
                cls.append(v["class_id"])
                ids.append(k)
                velocities.append(v["velocity"].unsqueeze(0))
                covariances.append(v["covariance"].unsqueeze(0))

        bboxs = torch.cat(bboxs) if len(bboxs) > 0 else torch.empty(0, 5)
        cls = torch.cat(cls) if len(cls) > 0 else torch.empty(0)
        ids = torch.tensor(ids).to(bboxs.device)  # type: ignore
        velocities = (
            torch.cat(velocities) if len(velocities) > 0 else torch.empty(0, 4)
        )
        covariances = (
            torch.cat(covariances)
            if len(covariances) > 0
            else torch.empty(0, 4)
        )
        return Boxes2D(bboxs, cls, ids), velocities, covariances

    def forward(  # type: ignore # pylint: disable=arguments-differ
        self, detections: Boxes2D, frame_id: int
    ) -> Boxes2D:
        """Process inputs, match detections with existing tracks."""
        print("#" * 100)
        print("A new frame:   frame = ", frame_id)
        print("#" * 100)
        _, inds = detections.boxes[:, -1].sort(descending=True)
        detections = detections[inds, :].to(torch.device("cpu"))

        # init ids container
        ids = torch.full((len(detections),), -1, dtype=torch.long)
        # match if buffer is not empty
        det_bboxes = detections.boxes[:, :-1]
        det_cls_ids = detections.class_ids
        # det_cls_unique = torch.unique(det_cls_ids)

        tracks_boxes2d, tracks_vel, tracks_cov = self.get_tracks(frame_id - 1)
        print("existing tracks ids:  ", self.tracks.keys())
        tracks_bboxes = tracks_boxes2d.boxes[:, :-1]
        tracks_ids = tracks_boxes2d.track_ids
        tracks_cls_ids = tracks_boxes2d.class_ids
        tracks_cls_unique = torch.unique(tracks_cls_ids)

        kalman_state = torch.cat(
            (xyxy_to_xyah(tracks_bboxes), tracks_vel), dim=1
        )
        for i, _ in enumerate(kalman_state):
            kalman_state[i], tracks_cov[i] = self.kf.predict(
                kalman_state[i], tracks_cov[i]
            )
        # tracks_bboxes = xyah_to_xyxy(kalman_state[:, :4])
        tracks_vel = kalman_state[:, 4:]

        updated_tracks_vels = dict()
        updated_tracks_covs = dict()
        # print("tracks_cls_ids:  ", tracks_cls_ids)
        # print("tracks_cls_unique:  ", tracks_cls_unique)
        # print("det_cls_ids:  ", det_cls_ids)

        for existing_cls in tracks_cls_unique:
            print("-" * 50)
            print("start matching for object class:  ", existing_cls)
            tracks_boxes_per_cls = tracks_bboxes[
                tracks_cls_ids == existing_cls
            ]
            tracks_indices_per_cls = torch.nonzero(
                tracks_cls_ids == existing_cls
            ).squeeze(1)
            det_boxes_per_cls = det_bboxes[det_cls_ids == existing_cls]
            det_indices_per_cls = torch.nonzero(
                det_cls_ids == existing_cls
            ).squeeze(1)
            print("tracks_indices_per_cls:  ", tracks_indices_per_cls)
            print("det_indices_per_cls:  ", det_indices_per_cls)

            matches, _, _ = self._match(
                tracks_boxes_per_cls,
                det_boxes_per_cls,
                tracks_indices_per_cls,
                det_indices_per_cls,
            )
            print("matched result:  ", matches)
            for matched_track_ind, matched_det_ind in matches:
                # print("matched_track_ind:  ", matched_track_ind)
                print("_" * 20)
                print("start updating detection indices: ", matched_det_ind)

                matched_kalman_state = kalman_state[matched_track_ind]
                matched_cov = tracks_cov[matched_track_ind]

                matched_det_bboxes = xyxy_to_xyah(
                    det_bboxes[matched_det_ind].unsqueeze(0)
                ).squeeze()
                updated_kalman_state, updated_track_cov = self.kf.update(
                    matched_kalman_state,
                    matched_cov,
                    matched_det_bboxes,
                )
                # print("updated_kalman_state  ", updated_kalman_state)
                updated_track_xyxy = xyah_to_xyxy(
                    updated_kalman_state[:4].unsqueeze(0)
                ).squeeze()
                updated_track_vel = updated_kalman_state[4:]
                updated_tracks_vels[
                    int(tracks_ids[matched_track_ind])
                ] = updated_track_vel
                updated_tracks_covs[
                    int(tracks_ids[matched_track_ind])
                ] = updated_track_cov
                detections.boxes[matched_det_ind, :-1] = updated_track_xyxy

                ids[matched_det_ind] = tracks_ids[matched_track_ind]

        new_inds = (ids == -1).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracks, self.num_tracks + num_news, dtype=torch.long
        )
        self.num_tracks += num_news
        print("updated detections tracking ids:  ", ids)

        self.update(
            ids, detections, frame_id, updated_tracks_vels, updated_tracks_covs
        )
        result, _, _ = self.get_tracks(frame_id)
        return result

    def _match(
        self,
        tracks_boxes: torch.Tensor,
        det_boxes: torch.Tensor,
        tracks_indices,
        det_indices,
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

        row_indices, col_indices = linear_assignment(iou_cost_matrix)
        # print("row_indices:  ", row_indices)
        # print("col_indices:  ", col_indices)
        matches, unmatched_tracks, unmatched_detections = [], [], []
        for col, detection_idx in enumerate(det_indices):
            if col not in col_indices:
                unmatched_detections.append(detection_idx)
        for row, track_idx in enumerate(tracks_indices):
            if row not in row_indices:
                unmatched_tracks.append(track_idx)
        for row, col in zip(row_indices, col_indices):
            # print("row, col : ", row, ",   ", col)
            track_idx = tracks_indices[row]
            detection_idx = det_indices[col]
            if iou_cost_matrix[row, col] > self.cfg.max_IOU_distance:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))
        return matches, unmatched_tracks, unmatched_detections

    def update(  # type: ignore # pylint: disable=arguments-differ
        self,
        ids: torch.Tensor,
        detections: Boxes2D,
        frame_id: int,
        updated_tracks_vels: Dict[int, torch.Tensor],
        updated_tracks_covs: Dict[int, torch.Tensor],
    ) -> None:
        """Update track memory using matched detections."""
        tracklet_inds = ids > -1
        # update memo
        for cur_id, det in zip(  # type: ignore
            ids[tracklet_inds], detections[tracklet_inds]
        ):
            cur_id = int(cur_id)
            if cur_id in self.tracks.keys():
                self.update_track(
                    cur_id,
                    det,
                    frame_id,
                    updated_tracks_vels[cur_id],
                    updated_tracks_covs[cur_id],
                )
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
        # print("update_track, ", "bbox: ", bbox, "   velocity:  ", velocity)
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
        # print("creat_track, bbox:  ", bbox)
        _, covariance = self.kf.initiate(
            xyxy_to_xyah(bbox[:-1].unsqueeze(0)).squeeze()
        )
        self.tracks[track_id] = dict(
            bbox=bbox,
            class_id=cls,
            last_frame=frame_id,
            velocity=torch.zeros_like(bbox[:-1]),
            covariance=covariance,
        )


class KalmanFilter(object):
    """Kalman Filter class.

    The 8-dimensional state space,
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    """

    def __init__(self):
        """Init."""
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = torch.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = torch.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = torch.zeros_like(mean_pos)
        mean = torch.cat([mean_pos, mean_vel])

        std = torch.Tensor(
            [
                2 * self._std_weight_position * measurement[3],
                2 * self._std_weight_position * measurement[3],
                1e-2,
                2 * self._std_weight_position * measurement[3],
                10 * self._std_weight_velocity * measurement[3],
                10 * self._std_weight_velocity * measurement[3],
                1e-5,
                10 * self._std_weight_velocity * measurement[3],
            ]
        )
        covariance = torch.diag(torch.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step."""
        std_pos = torch.Tensor(
            [
                self._std_weight_position * mean[3],
                self._std_weight_position * mean[3],
                1e-2,
                self._std_weight_position * mean[3],
            ]
        )
        std_vel = torch.Tensor(
            [
                self._std_weight_velocity * mean[3],
                self._std_weight_velocity * mean[3],
                1e-5,
                self._std_weight_velocity * mean[3],
            ]
        )
        motion_cov = torch.diag(torch.square(torch.cat((std_pos, std_vel))))

        mean = torch.matmul(self._motion_mat, mean)
        covariance = (
            torch.matmul(
                self._motion_mat, torch.matmul(covariance, self._motion_mat.T)
            )
            + motion_cov
        )

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space."""
        std = torch.Tensor(
            [
                10 * self._std_weight_position * mean[3],
                10 * self._std_weight_position * mean[3],
                1e-1,
                10 * self._std_weight_position * mean[3],
            ]
        )
        innovation_cov = torch.diag(torch.square(std))

        mean = torch.matmul(self._update_mat, mean)
        covariance = torch.matmul(
            self._update_mat, torch.matmul(covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step."""
        projected_mean, projected_cov = self.project(mean, covariance)
        projected_cov = projected_cov.numpy()
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            torch.matmul(covariance, self._update_mat.T).numpy().T,
            check_finite=False,
        ).T
        kalman_gain = torch.from_numpy(kalman_gain)
        innovation = measurement - projected_mean

        new_mean = mean + torch.matmul(innovation, kalman_gain.T)
        projected_cov = torch.from_numpy(projected_cov)
        new_covariance = covariance - torch.matmul(
            kalman_gain, torch.matmul(projected_cov, kalman_gain.T)
        )

        return new_mean, new_covariance
