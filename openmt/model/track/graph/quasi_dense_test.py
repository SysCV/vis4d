"""Test cases for quasi dense tracking graph construction."""
import unittest

import torch

from openmt.common.bbox.utils import compute_iou
from openmt.struct import Boxes2D
from openmt.unittest.utils import generate_dets

from .base import TrackGraphConfig
from .quasi_dense import QDTrackGraph


class TestQDTrackGraph(unittest.TestCase):
    """Test cases for quasi-dense tracking graph construction."""

    def test_get_tracks(self) -> None:
        """Testcase for get tracks method."""
        tracker = QDTrackGraph(
            TrackGraphConfig(type="qdtrack", keep_in_memory=3)
        )

        h, w, num_dets = 128, 128, 64
        detections = generate_dets(h, w, num_dets)
        embeddings = torch.rand(num_dets, 128)

        for i in range(num_dets):
            tracker.create_track(i, detections[i], embeddings[i], 0)

        boxes, embeds = tracker.get_tracks(torch.device("cpu"), frame_id=0)
        self.assertTrue(len(boxes) == len(embeds) == num_dets)

        for i in range(num_dets // 2):
            tracker.update_track(i, detections[i], embeddings[i], 1)

        boxes, embeds = tracker.get_tracks(torch.device("cpu"), frame_id=1)
        self.assertTrue(len(boxes) == len(embeds) == num_dets // 2)

        boxes, embeds = tracker.get_tracks(torch.device("cpu"), frame_id=2)
        self.assertTrue(len(boxes) == len(embeds) == 0)

    def test_track(self) -> None:
        """Testcase for tracking function."""
        tracker = QDTrackGraph(
            TrackGraphConfig(type="qdtrack", keep_in_memory=3)
        )

        h, w, num_dets = 128, 128, 64
        detections = generate_dets(h, w, num_dets)
        embeddings = torch.rand(num_dets, 128)

        # feed same detections & embeddings --> should be matched to self
        result_t0 = tracker(detections, 0, embeddings)
        result_t1 = tracker(detections, 1, embeddings)
        result_t2 = tracker(detections, 2, embeddings)

        empty_det, empty_emb = Boxes2D(torch.empty(0, 5)), torch.empty(0, 128)
        for i in range(tracker.cfg.keep_in_memory + 1):
            result_final = tracker(empty_det, 3 + i, empty_emb)

        self.assertTrue(tracker.empty)
        self.assertEqual(len(result_final), 0)

        # check if matching is correct
        for t0, t1, t2 in zip(
            result_t0.track_ids, result_t1.track_ids, result_t2.track_ids
        ):
            self.assertTrue(t0 == t1 == t2)

        # check if all tracks have scores >= threshold
        for res in [result_t0, result_t1, result_t2]:
            self.assertTrue(
                (res.boxes[:, -1] >= tracker.cfg.obj_score_thr).all()
            )

            # check if tracks do not overlap too much
            ious = compute_iou(res, res) - torch.eye(len(res.boxes))
            self.assertTrue((ious <= tracker.cfg.nms_class_iou_thr).all())
