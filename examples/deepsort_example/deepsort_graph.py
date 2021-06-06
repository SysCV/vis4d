"""Track graph of deep SORT."""
from typing import Dict, Optional, Tuple
import torch

from kalman_filter import KalmanFilter
from match import match
from openmt.struct import Boxes2D
from openmt.model.track.graph import BaseTrackGraph, TrackGraphConfig


class DeepSORTTrackGraphConfig(TrackGraphConfig):
    """deep SORT graph config."""

    featurenet_weight_path: str = "/home/yinjiang/systm/examples/deepsort_example/checkpoint/original_ckpt.t7"
    keep_in_memory: int = 1  # threshold for keeping occluded objects in memory
    max_IOU_distance: float = 0.7


class DeepSORTTrackGraph(BaseTrackGraph):
    """deep SORT tracking logic."""

    def __init__(self, cfg: TrackGraphConfig) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg = DeepSORTTrackGraphConfig(**cfg.dict())
        self.kf = KalmanFilter()

    def get_tracks(
        self, frame_id: Optional[int] = None
    ) -> Tuple[Boxes2D, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get active tracks at given frame.

        If frame_id is None, return all tracks in memory.
        """
        bboxs, cls, ids, velocities, covariances, features = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for k, v in self.tracks.items():
            if frame_id is None or v["last_frame"] == frame_id:
                bboxs.append(v["bbox"].unsqueeze(0))
                cls.append(v["class_id"])
                ids.append(k)
                velocities.append(v["velocity"].unsqueeze(0))
                covariances.append(v["covariance"].unsqueeze(0))
                features.append(v["feature"].unsqueeze(0))
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
        features = (
            torch.cat(features) if len(features) > 0 else torch.empty(0, 128)
        )
        return Boxes2D(bboxs, cls, ids), velocities, covariances, features

    def forward(  # type: ignore # pylint: disable=arguments-differ
        self, detections: Boxes2D, frame_id: int, det_features: torch.Tensor
    ) -> Boxes2D:
        """Process inputs, match detections with existing tracks.

        image tensor is shape (C_1, ..., C_K, H, W) where K >= 1.
        """
        if len(detections) == 0:
            result, _, _, _ = self.get_tracks(frame_id)
            return result
        # print("#" * 100)
        # print("A new frame:   frame = ", frame_id)
        # print("#" * 100)

        _, inds = detections.boxes[:, -1].sort(descending=True)
        detections = detections[inds, :].to(torch.device("cpu"))
        det_features = det_features[inds, :].to(torch.device("cpu"))
        # init ids container
        ids = torch.full((len(detections),), -1, dtype=torch.long)
        # match if buffer is not empty
        det_bboxes = detections.boxes[:, :-1]
        det_cls_ids = detections.class_ids
        # det_cls_unique = torch.unique(det_cls_ids)

        (
            tracks_boxes2d,
            tracks_vel,
            tracks_cov,
            tracks_features,
        ) = self.get_tracks(frame_id - 1)
        # print("existing tracks ids:  ", self.tracks.keys())
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
        # comment this line to not using prediction
        tracks_bboxes = xyah_to_xyxy(kalman_state[:, :4])
        predictions = Boxes2D(tracks_bboxes, tracks_cls_ids, tracks_ids)

        tracks_vel = kalman_state[:, 4:]

        updated_tracks_vels = dict()
        updated_tracks_covs = dict()
        # print("tracks_cls_ids:  ", tracks_cls_ids)
        # print("tracks_cls_unique:  ", tracks_cls_unique)
        # print("det_cls_ids:  ", det_cls_ids)

        for existing_cls in tracks_cls_unique:
            # print("-" * 50)
            # print("start matching for object class:  ", existing_cls)
            tracks_boxes_per_cls = tracks_bboxes[
                tracks_cls_ids == existing_cls
            ]
            tracks_indices_per_cls = torch.nonzero(
                tracks_cls_ids == existing_cls
            ).squeeze(1)
            tracks_features_per_cls = tracks_features[
                tracks_cls_ids == existing_cls
            ]
            det_boxes_per_cls = det_bboxes[det_cls_ids == existing_cls]
            det_indices_per_cls = torch.nonzero(
                det_cls_ids == existing_cls
            ).squeeze(1)
            det_features_per_cls = det_features[det_cls_ids == existing_cls]
            # print("tracks_indices_per_cls:  ", tracks_indices_per_cls)
            # print("det_indices_per_cls:  ", det_indices_per_cls)

            matches, _, _ = match(
                tracks_boxes_per_cls,
                det_boxes_per_cls,
                tracks_indices_per_cls,
                det_indices_per_cls,
                det_features_per_cls,
                tracks_features_per_cls,
            )
            # print("matched result:  ", matches)
            for matched_track_ind, matched_det_ind in matches:
                # print("matched_track_ind:  ", matched_track_ind)
                # print("_" * 20)
                # print("start updating detection indices: ", matched_det_ind)

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

                # comment this line to not use corrected bbox
                # detections.boxes[matched_det_ind, :-1] = updated_track_xyxy

                ids[matched_det_ind] = tracks_ids[matched_track_ind]

        new_inds = (ids == -1).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracks, self.num_tracks + num_news, dtype=torch.long
        )
        self.num_tracks += num_news
        # print("updated detections tracking ids:  ", ids)

        self.update(
            ids,
            detections,
            det_features,
            frame_id,
            updated_tracks_vels,
            updated_tracks_covs,
        )
        result, _, _, _ = self.get_tracks(frame_id)
        return result

    def update(  # type: ignore # pylint: disable=arguments-differ
        self,
        ids: torch.Tensor,
        detections: Boxes2D,
        det_features: torch.Tensor,
        frame_id: int,
        updated_tracks_vels: Dict[int, torch.Tensor],
        updated_tracks_covs: Dict[int, torch.Tensor],
    ) -> None:
        """Update track memory using matched detections."""
        tracklet_inds = ids > -1
        # update memo
        for cur_id, det, det_feature in zip(  # type: ignore
            ids[tracklet_inds],
            detections[tracklet_inds],
            det_features[tracklet_inds],
        ):
            cur_id = int(cur_id)
            if cur_id in self.tracks.keys():
                self.update_track(
                    cur_id,
                    det,
                    frame_id,
                    updated_tracks_vels[cur_id],
                    updated_tracks_covs[cur_id],
                    det_feature,
                )
            else:
                self.create_track(cur_id, det, frame_id, det_feature)

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
        feature: torch.Tensor,
    ) -> None:
        """Update a specific track with a new detection."""
        bbox, cls = detection.boxes[0], detection.class_ids[0]
        # print("update_track, ", "bbox: ", bbox, "   velocity:  ", velocity)
        self.tracks[track_id]["bbox"] = bbox
        self.tracks[track_id]["last_frame"] = frame_id
        self.tracks[track_id]["class_id"] = cls
        self.tracks[track_id]["velocity"] = velocity
        self.tracks[track_id]["covariance"] = covariance
        self.tracks[track_id]["feature"] = feature

    def create_track(
        self,
        track_id: int,
        detection: Boxes2D,
        frame_id: int,
        feature: torch.Tensor,
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
            feature=feature,
        )


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
