"""Track graph of deep SORT."""
import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch

from vis4d.common.bbox.utils import bbox_iou
from vis4d.struct import Boxes2D, InputSample, LabelInstances, LossesType

from ..motion import KalmanFilter
from .base import BaseTrackGraph, TrackGraphConfig
from .deep_sort_utils import (
    NearestNeighborDistanceMetric,
    gate_cost_matrix,
    matching_cascade,
    min_cost_matching,
    tlbr_to_xyah,
    xyah_to_tlbr,
)


class DeepSORTTrackGraphConfig(TrackGraphConfig):
    """deep SORT graph config."""

    min_confidence: float = 0.3
    max_cosine_distance: float = 0.2
    max_age: int = 70
    n_init: int = 1
    nn_budget: int = 100
    max_iou_distance: float = 0.7
    nms_max_overlap: float = 0.5
    kf_parameters: Dict[str, List[List[float]]]


class DeepSORTTrackGraph(BaseTrackGraph):
    """deep SORT tracking logic."""

    def __init__(self, cfg: TrackGraphConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg = DeepSORTTrackGraphConfig(**cfg.dict())
        self.max_iou_distance = self.cfg.max_iou_distance
        self.max_age = self.cfg.max_age
        self.n_init = self.cfg.n_init
        self.metric = NearestNeighborDistanceMetric(
            self.cfg.max_cosine_distance, self.cfg.nn_budget
        )
        kf_motion_mat = torch.eye(8, 8)
        for i in range(4):
            kf_motion_mat[i, 4 + i] = 1.0
        kf_update_mat = torch.eye(4, 8)
        self.kf = torch.nn.ModuleDict()
        self.num_tracks = 0
        for class_id in range(len(self.cfg.kf_parameters["cov_motion_Q"])):
            self.kf[str(class_id)] = KalmanFilter(
                kf_motion_mat,
                kf_update_mat,
                torch.diag(
                    torch.tensor(
                        self.cfg.kf_parameters["cov_motion_Q"][class_id]
                    )
                ),
                torch.diag(
                    torch.tensor(
                        self.cfg.kf_parameters["cov_project_R"][class_id]
                    )
                ),
                torch.diag(
                    torch.tensor(self.cfg.kf_parameters["cov_P0"][class_id])
                ),
            )

        self.reset()

    def reset(self) -> None:
        """Reset tracks."""
        self.num_tracks = 0
        self.tracks = {}  # type: ignore

    def get_tracks(self) -> Boxes2D:
        """Get active tracks at current frame."""
        track_boxes = []
        class_ids = []
        track_ids = []
        for track_id, track in self.tracks.items():
            if track["time_since_update"] >= 1:
                continue
            x1, y1, x2, y2 = xyah_to_tlbr(track["mean"][:4])
            conf = track["confidence"]
            track_boxes.append(
                torch.tensor([x1, y1, x2, y2, conf]).unsqueeze(0)
            )
            class_ids.append(track["class_id"])
            track_ids.append(track_id)

        track_boxes = (
            torch.cat(track_boxes)
            if len(track_boxes) > 0
            else torch.empty((0, 5))
        )
        class_ids = (
            torch.tensor(class_ids) if len(class_ids) > 0 else torch.empty(0)
        )
        track_ids = (
            torch.tensor(track_ids) if len(track_ids) > 0 else torch.empty(0)
        )
        return Boxes2D(track_boxes, class_ids, track_ids)

    def forward_train(
        self,
        inputs: List[InputSample],
        predictions: List[LabelInstances],
        targets: Optional[List[LabelInstances]],
        **kwargs: List[torch.Tensor],
    ) -> LossesType:
        """Forward of QDTrackGraph in training stage."""
        raise NotImplementedError

    def forward_test(
        self,
        inputs: InputSample,
        predictions: LabelInstances,
        embeddings: Optional[torch.Tensor] = None,
        **kwargs: torch.Tensor,
    ) -> LabelInstances:
        """Process inputs, match detections with existing tracks."""
        detections = predictions.boxes2d[0].clone()
        frame_id = inputs.metadata[0].frameIndex
        assert (
            frame_id is not None
        ), "Couldn't find current frame index in InputSample metadata!"

        # reset graph at begin of sequence
        if frame_id == 0:
            self.reset()
        det_boxes = detections.boxes
        confidences = det_boxes[:, -1]
        select_idx = confidences >= self.cfg.min_confidence
        detections_selected = detections[select_idx]
        if embeddings is not None:
            embeddings = embeddings[0].clone()
            det_features_selected = embeddings[select_idx]

        self.predict()
        if embeddings is not None:
            self.update(detections_selected, det_features_selected)
        else:
            self.update(detections_selected)
        output = self.get_tracks()
        result = copy.deepcopy(predictions)
        result.boxes2d[0] = output
        return result

    def predict(self) -> None:
        """Propagate all tracklet one time step forward.

        This function should be called once every time step, before `update`.
        """
        for _, track in self.tracks.items():
            class_id = track["class_id"]
            mean, covariance = self.kf[str(class_id)].predict(
                track["mean"], track["covariance"]
            )
            track["mean"] = mean
            track["covariance"] = covariance
            track["age"] += 1
            track["time_since_update"] += 1

    def update(  # pylint: disable=arguments-differ
        self, detections: Boxes2D, det_features: torch.tensor = None
    ) -> None:
        """Perform association and track management."""
        cls_detidx_mapping = defaultdict(list)
        for i, class_id in enumerate(detections.class_ids):
            cls_detidx_mapping[int(class_id)].append(i)
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = [], [], []
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
        # Update track set.
        for track_id, det_idx in matches:
            assert det_idx not in matched_det_set
            matched_det_set.add(det_idx)

            track = self.tracks[track_id]
            class_id = track["class_id"]
            # kf measurement update step and update the feature cache.
            new_pos = tlbr_to_xyah(detections.boxes[det_idx][:4])
            track["mean"], track["covariance"] = self.kf[str(class_id)].update(
                track["mean"],
                track["covariance"],
                new_pos,
            )
            track["mean"][:4] = new_pos
            track["confidence"] = float(detections.boxes[det_idx][4])
            if det_features is not None:
                track["features"].append(det_features[det_idx])
            track["hits"] += 1
            track["time_since_update"] = 0
            if track["state"] == "Tentative" and track["hits"] >= self._n_init:
                track["state"] = "Confirmed"  # pragma: no cover

        # mark unmatched tracks to 'delete' in certain cases
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            if track["state"] == "Tentative":
                track["state"] = "Deleted"  # pragma: no cover
            elif track["time_since_update"] > self.max_age:
                track["state"] = "Deleted"  # pragma: no cover

        for det_idx in unmatched_detections:
            det_box = tlbr_to_xyah(detections.boxes[det_idx][:4])
            confidence = float(detections.boxes[det_idx][4])
            class_id = int(detections.class_ids[det_idx])
            if det_features is not None:
                self.create_track(
                    det_box, class_id, confidence, det_features[det_idx]
                )
            else:
                self.create_track(det_box, class_id, confidence)
        # may cut this step and merge with the marking process above
        for t_id in list(self.tracks.keys()):
            if self.tracks[t_id]["state"] == "Deleted":
                self.tracks.pop(t_id)  # pragma: no cover

        for unmatched_det in unmatched_detections:
            assert unmatched_det not in unmatched_det_set
            unmatched_det_set.add(unmatched_det)
        assert len(matched_det_set & unmatched_det_set) == 0
        assert len(matched_det_set | unmatched_det_set) == len(detections)

        if det_features is not None:
            # Update distance metric.
            active_targets = [
                t_id
                for t_id, t in self.tracks.items()
                if t["state"] == "Confirmed"
            ]
            features, targets = [], []
            for t_id, track in self.tracks.items():
                if not track["state"] == "Confirmed":
                    continue  # pragma: no cover
                features += track["features"]
                targets += [t_id for _ in track["features"]]
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
        """Matching."""
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            t_id
            for t_id, t in self.tracks.items()
            if t["state"] == "Confirmed" and t["class_id"] == class_id
        ]
        unconfirmed_tracks = [
            t_id
            for t_id, t in self.tracks.items()
            if not t["state"] == "Confirmed" and t["class_id"] == class_id
        ]

        if det_features is not None:

            def gated_metric(
                tracks: Dict[int, Dict[str, Union[int, float, torch.Tensor]]],
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
                cost_matrix = gate_cost_matrix(
                    self.kf[str(class_id)],
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
            ) = matching_cascade(
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
                if self.tracks[k]["time_since_update"] == 1
            ]
            unmatched_tracks_a = [
                k
                for k in unmatched_tracks_a
                if self.tracks[k]["time_since_update"] != 1
            ]

            bbox = torch.empty((0, 5)).to(detections.device)

            for _, track_id in enumerate(iou_track_candidates):
                bbox_t = xyah_to_tlbr(self.tracks[track_id]["mean"][:4])
                conf = self.tracks[track_id]["confidence"]
                bbox_t = torch.cat(
                    (bbox_t, torch.tensor([conf]).to(bbox_t.device))
                ).unsqueeze(0)
                bbox = torch.cat((bbox, bbox_t), dim=0)
            iou_track_candidates_box2d = Boxes2D(bbox)
            unmatch_detections_box2d = detections[unmatched_detections]
            iou_res = bbox_iou(
                iou_track_candidates_box2d, unmatch_detections_box2d
            )
            iou_cost_matrix = torch.ones_like(iou_res) - iou_res

            (
                matches_b,
                unmatched_tracks_b,
                unmatched_detections,
            ) = min_cost_matching(
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
                bbox_t = xyah_to_tlbr(self.tracks[track_id]["mean"][:4])
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
            ) = min_cost_matching(
                iou_cost_matrix,
                self.max_iou_distance,
                self.tracks,
                detections,
                confirmed_tracks,
                detection_indices,
            )

        return matches, unmatched_tracks, unmatched_detections

    def create_track(
        self,
        det_box: torch.tensor,
        class_id: int,
        confidence: float,
        det_feature: torch.tensor = None,
    ) -> None:
        """Create a new track from a models."""
        mean, covariance = self.kf[str(class_id)].initiate(det_box)
        if det_feature is not None:
            track = {
                "mean": mean,
                "covariance": covariance,
                "confidence": confidence,
                "class_id": class_id,
                "features": [det_feature],
                "state": "Confirmed",
                "hits": 1,
                "age": 1,
                "time_since_update": 0,
            }
        else:
            track = {
                "mean": mean,
                "covariance": covariance,
                "confidence": confidence,
                "class_id": class_id,
                "state": "Confirmed",
                "hits": 1,
                "age": 1,
                "time_since_update": 0,
            }
        self.tracks[self.num_tracks] = track
        self.num_tracks += 1
