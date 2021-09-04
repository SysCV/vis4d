"""Track graph of deep SORT."""
from collections import defaultdict
from typing import List

import numpy as np
import torch

from vist.struct import Boxes2D, NDArrayF64
from ...deepsort_example.iou_matching import iou_cost
from ...deepsort_example.linear_assignment import (
    gate_cost_matrix,
    min_cost_matching,
    matching_cascade,
)
from ...deepsort_example.detection import Detection
from ...deepsort_example.kalman_filter import KalmanFilter
from ...deepsort_example.nn_matching import NearestNeighborDistanceMetric
from ...deepsort_example.track import Track
from .base import BaseTrackGraph, TrackGraphConfig

# from ...deepsort_example.preprocessing import non_max_suppression


def tlwh_to_tlbr(bbox_tlwh: NDArrayF64):
    """Convert a single bbox from tlwh to tlbr."""
    x1, y1, w, h = bbox_tlwh
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2


def tlbr_to_tlwh(bbox_tlbr: NDArrayF64):
    """Convert tlbr boxes to tlwh.

    bbox_tlbr: torch.FloatTensor: (N, 4) where each entry is defined by
    [x1, y1, x2, y2]
    """
    bbox_tlwh = bbox_tlbr.copy()
    bbox_tlwh[:, 2] = bbox_tlbr[:, 2] - bbox_tlbr[:, 0]
    bbox_tlwh[:, 3] = bbox_tlbr[:, 3] - bbox_tlbr[:, 1]
    return bbox_tlwh


class DeepSORTTrackGraphConfig(TrackGraphConfig):
    """deep SORT graph config."""

    min_confidence: float = 0.3
    metric = "cosine"
    max_cosine_distance = 0.2
    max_age = 70
    n_init = 1
    nn_budget = 100
    max_iou_distance = 0.7
    nms_max_overlap = 0.5


class DeepSORTTrackGraph(BaseTrackGraph):
    """deep SORT tracking logic."""

    def __init__(self, cfg: TrackGraphConfig) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg = DeepSORTTrackGraphConfig(**cfg.dict())
        self.kf = KalmanFilter()
        self.max_iou_distance = self.cfg.max_iou_distance
        self.max_age = self.cfg.max_age
        self.n_init = self.cfg.n_init
        self.metric = NearestNeighborDistanceMetric(
            self.cfg.metric, self.cfg.max_cosine_distance, self.cfg.nn_budget
        )
        self._next_id = 1
        self.tracks = []  # type: ignore
        self.reset()

    def reset(self) -> None:
        """Reset tracks."""
        self._next_id = 1
        self.tracks = []  # type: ignore

    def get_output(self) -> Boxes2D:
        """Get active tracks at given frame.

        If frame_id is None, return all tracks in memory.
        """
        track_boxes = []
        class_ids = []
        track_ids = []
        for track in self.tracks:
            # if not track.is_confirmed() or track.time_since_update > 1:
            #     continue
            if track.time_since_update >= 1:
                continue
            x1, y1, x2, y2 = track.to_tlbr()
            conf = track.confidence
            track_boxes.append(
                torch.tensor([x1, y1, x2, y2, conf]).unsqueeze(0)
            )
            class_ids.append(track.class_id)
            track_ids.append(track.track_id)

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

    def forward(  # type: ignore # pylint: disable=arguments-differ
        self,
        detections: Boxes2D,
        frame_id: int,
        det_features: torch.Tensor,
    ) -> Boxes2D:
        """Process inputs, match detections with existing tracks."""
        det_boxes = detections.boxes.to(torch.device("cpu")).numpy()
        bbox_tlbr = det_boxes[:, :-1]
        confidences = det_boxes[:, -1]
        class_ids = detections.class_ids.to(torch.device("cpu"))
        det_features = det_features.to(torch.device("cpu")).numpy()
        bbox_tlwh = tlbr_to_tlwh(bbox_tlbr)
        dets = [
            Detection(bbox_tlwh[i], conf, int(class_id), det_features[i])
            for i, (conf, class_id) in enumerate(zip(confidences, class_ids))
            if conf >= self.cfg.min_confidence
        ]
        # # run on non-maximum supression, don't use it on good detection
        # boxes = np.array([d.tlwh for d in dets])
        # scores = np.array([d.confidence for d in dets])
        # indices = non_max_suppression(
        #     boxes, self.cfg.nms_max_overlap, scores
        # )  # pylint:disable=line-too-long
        # dets = [dets[i] for i in indices]

        self.predict()
        self.update(dets)

        output = self.get_output()
        return output

    def predict(self):
        """Propagate all tracklet one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections: List[Detection]):  # type: ignore # pylint: disable=arguments-differ
        """Perform association and track management."""
        cls_detidx_mapping = defaultdict(list)
        for i, det in enumerate(detections):
            cls_detidx_mapping[det.class_id].append(i)
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = [], [], []
        for class_id, detidx in cls_detidx_mapping.items():
            (
                matches_cls,
                unmatched_tracks_cls,
                unmatched_detections_cls,
            ) = self._match(detections, detidx, class_id)
            matches.extend(matches_cls)
            unmatched_tracks.extend(unmatched_tracks_cls)
            unmatched_detections.extend(unmatched_detections_cls)

        matched_det_set = set()
        unmatched_det_set = set()
        # Update track set.
        for track_idx, detection_idx in matches:
            assert detection_idx not in matched_det_set
            matched_det_set.add(detection_idx)
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [
            t for t in self.tracks if not t.is_deleted()
        ]  # type:ignore
        for unmatched_det in unmatched_detections:
            assert unmatched_det not in unmatched_det_set
            unmatched_det_set.add(unmatched_det)
        assert len(matched_det_set & unmatched_det_set) == 0
        assert len(matched_det_set | unmatched_det_set) == len(detections)

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def _match(self, detections: List[Detection], detection_indices, class_id):
        """Matching."""

        def gated_metric(tracks, dets, track_indices, detection_indices):
            """Calculate cost matrix."""
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            # calculate cost matrix using deep feature
            cost_matrix = self.metric.distance(features, targets)
            # use mahalanobis distance to gate cost matrix
            cost_matrix = gate_cost_matrix(
                self.kf,
                cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
            )

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i
            for i, t in enumerate(self.tracks)
            if t.is_confirmed() and t.class_id == class_id
        ]
        unconfirmed_tracks = [
            i
            for i, t in enumerate(self.tracks)
            if not t.is_confirmed() and t.class_id == class_id
        ]

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
            confirmed_tracks,
            detection_indices,
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU
        iou_track_candidates = unconfirmed_tracks + [
            k
            for k in unmatched_tracks_a
            if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k
            for k in unmatched_tracks_a
            if self.tracks[k].time_since_update != 1
        ]
        (
            matches_b,
            unmatched_tracks_b,
            unmatched_detections,
        ) = min_cost_matching(
            iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        """Initiate a track."""
        mean, covariance = self.kf.initiate(
            detection.to_xyah(), detection.class_id
        )
        confidence, class_id = detection.confidence, detection.class_id
        self.tracks.append(
            Track(
                mean,
                covariance,
                confidence,
                class_id,
                self._next_id,
                self.n_init,
                self.max_age,
                detection.feature,
            )
        )
        self._next_id += 1
