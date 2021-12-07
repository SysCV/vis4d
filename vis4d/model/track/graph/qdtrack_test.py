"""Test cases for quasi dense tracking graph construction."""
import unittest

import torch
from scalabel.label.typing import Frame

from vis4d.common.bbox.utils import bbox_iou
from vis4d.struct import Boxes2D, Images, InputSample, LabelInstances
from vis4d.unittest.utils import generate_dets

from .base import TrackGraphConfig
from .qdtrack import QDTrackGraph


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
        sample = InputSample(
            [Frame(name="myframe", frameIndex=0)],
            Images(torch.empty(1, 128, 128, 3), image_sizes=[(128, 128)]),
        )
        dets = LabelInstances([generate_dets(h, w, num_dets)])
        embeddings = [torch.rand(num_dets, 128)]

        # feed same detections & embeddings --> should be matched to self
        result_t0 = tracker(sample, dets, embeddings=embeddings).boxes2d[0]
        sample.metadata[0].frameIndex = 1
        result_t1 = tracker(sample, dets, embeddings=embeddings).boxes2d[0]
        sample.metadata[0].frameIndex = 1
        result_t2 = tracker(sample, dets, embeddings=embeddings).boxes2d[0]

        empty_det = LabelInstances([Boxes2D(torch.empty(0, 5))])
        empty_emb = [torch.empty(0, 128)]
        for i in range(tracker.cfg.keep_in_memory + 1):
            sample.metadata[0].frameIndex = 3 + i
            result_final = tracker(sample, empty_det, embeddings=empty_emb)

        self.assertTrue(tracker.empty)
        self.assertEqual(len(result_final.boxes2d[0]), 0)

        # check if matching is correct
        t0, t1, t2 = (
            result_t0.track_ids.sort()[0],
            result_t1.track_ids.sort()[0],
            result_t2.track_ids.sort()[0],
        )
        self.assertTrue((t0 == t1).all() and (t1 == t2).all())

        # check if all tracks have scores >= threshold
        for res in [result_t0, result_t1, result_t2]:
            self.assertTrue(
                (res.boxes[:, -1] >= tracker.cfg.obj_score_thr).all()
            )

            # check if tracks do not overlap too much
            ious = bbox_iou(res, res) - torch.eye(len(res.boxes))
            self.assertTrue((ious <= tracker.cfg.nms_class_iou_thr).all())
