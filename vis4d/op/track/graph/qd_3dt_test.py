"""Test cases for QD3DT tracking graph construction."""
import unittest

import torch
from scalabel.label.typing import Frame

from vis4d.common_to_revise.bbox.utils import bbox_iou
from vis4d.struct_to_revise import (
    Boxes2D,
    Boxes3D,
    Images,
    InputSample,
    LabelInstances,
)
from vis4d.unittest.utils import generate_dets, generate_dets3d

from .qd_3dt import QD3DTrackGraph


class TestQD3DTrackGraph(unittest.TestCase):
    """Test cases for QD3DT tracking graph construction."""

    tracker = QD3DTrackGraph(keep_in_memory=3)

    def test_get_tracks(self) -> None:
        """Testcase for get tracks method."""
        h, w, num_dets = 128, 128, 64
        detections = generate_dets(h, w, num_dets)
        detections_3d = generate_dets3d(num_dets)
        embeddings = torch.rand(num_dets, 128)

        for i in range(num_dets):
            self.tracker.create_track(
                i, detections[i], detections_3d[i], embeddings[i], 0
            )

        (
            boxes2d,
            boxes3d,
            embeds,
            _,
            _,
        ) = self.tracker.get_tracks(torch.device("cpu"), frame_id=0)
        self.assertTrue(
            len(boxes2d) == len(boxes3d) == len(embeds) == num_dets
        )

        for i in range(num_dets // 2):
            self.tracker.update_track(
                i, detections[i], detections_3d[i], embeddings[i], 1
            )

        (
            boxes2d,
            boxes3d,
            embeds,
            _,
            _,
        ) = self.tracker.get_tracks(torch.device("cpu"), frame_id=1)
        self.assertTrue(
            len(boxes2d) == len(boxes3d) == len(embeds) == num_dets // 2
        )

        (
            boxes2d,
            boxes3d,
            embeds,
            _,
            _,
        ) = self.tracker.get_tracks(torch.device("cpu"), frame_id=2)
        self.assertTrue(len(boxes2d) == len(boxes3d) == len(embeds) == 0)

    def test_track(self) -> None:
        """Testcase for tracking function."""
        h, w, num_dets = 128, 128, 64
        sample = InputSample(
            [Frame(name="myframe", frameIndex=0)],
            Images(torch.empty(1, 128, 128, 3), image_sizes=[(128, 128)]),
        )
        predictions = LabelInstances(
            [generate_dets(h, w, num_dets)],
            [generate_dets3d(num_dets)],
        )
        embeddings = [torch.rand(num_dets, 128)]

        # feed same detections & embeddings --> should be matched to self
        result_t0 = self.tracker(sample, predictions, embeddings=embeddings)
        sample.metadata[0].frameIndex = 1

        result_t1 = self.tracker(sample, predictions, embeddings=embeddings)
        sample.metadata[0].frameIndex = 1
        result_t2 = self.tracker(sample, predictions, embeddings=embeddings)

        empty_predictions = LabelInstances(
            [Boxes2D.empty("cpu")], [Boxes3D.empty("cpu")]
        )
        empty_emb = [torch.empty(0, 128)]
        for i in range(self.tracker.keep_in_memory + 1):
            sample.metadata[0].frameIndex = 3 + i
            result_final = self.tracker(
                sample, empty_predictions, embeddings=empty_emb
            )

        self.assertTrue(self.tracker.empty)
        self.assertEqual(len(result_final.boxes3d[0]), 0)

        # check if matching is correct
        t0, t1, t2 = (
            result_t0.boxes3d[0].track_ids.sort()[0],
            result_t1.boxes3d[0].track_ids.sort()[0],
            result_t2.boxes3d[0].track_ids.sort()[0],
        )
        self.assertTrue((t0 == t1).all() and (t1 == t2).all())

        # check if all tracks have scores >= threshold
        for res in [result_t0, result_t1, result_t2]:
            self.assertTrue(
                (
                    res.boxes3d[0].boxes[:, -1] >= self.tracker.obj_score_thr
                ).all()
            )

            # check if tracks do not overlap too much
            ious = bbox_iou(res.boxes2d[0], res.boxes2d[0]) - torch.eye(
                len(res.boxes2d[0].boxes)
            )
            self.assertTrue((ious <= self.tracker.nms_class_iou_thr).all())
