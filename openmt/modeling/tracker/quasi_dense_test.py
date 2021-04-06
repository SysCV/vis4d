"""Test cases for quasi dense tracking config."""
import unittest
from typing import Tuple

import torch

from openmt.core.bbox.utils import compute_iou
from openmt.struct import Boxes2D

from .base import TrackLogicConfig
from .quasi_dense import QDEmbeddingTracker


class TestQDTracker(unittest.TestCase):
    """Test cases for quasi-dense tracking component."""

    def test_track(self) -> None:
        """Testcase for tracking function."""
        tracker = QDEmbeddingTracker(
            TrackLogicConfig(type="QDEmbeddingTrackerConfig", keep_in_memory=3)
        )

        h, w, num_dets = 128, 128, 10
        detections, embeddings = generate_dets(h, w, num_dets)

        # feed same detections & embeddings --> should be matched to self
        result_t0 = tracker(detections, embeddings, 0)
        result_t1 = tracker(detections, embeddings, 1)
        result_t2 = tracker(detections, embeddings, 2)

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


def generate_dets(
    height: int, width: int, num_dets: int
) -> Tuple[Boxes2D, torch.Tensor]:
    """Create random detections & embeddings."""
    rand_max = torch.repeat_interleave(
        torch.tensor([[width, height, width, height, 1.0]]), num_dets, dim=0
    )
    box_tensor = torch.rand(num_dets, 5) * rand_max
    sorted_xy = [
        box_tensor[:, [0, 2]].sort(dim=-1)[0],
        box_tensor[:, [1, 3]].sort(dim=-1)[0],
    ]
    box_tensor[:, :4] = torch.cat(
        [
            sorted_xy[0][:, 0:1],
            sorted_xy[1][:, 0:1],
            sorted_xy[0][:, 1:2],
            sorted_xy[1][:, 1:2],
        ],
        dim=-1,
    )
    dets = Boxes2D(box_tensor, torch.zeros(num_dets))
    embeds = torch.rand(num_dets, 128)
    return dets, embeds
