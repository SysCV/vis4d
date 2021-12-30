"""Test cases for deepsort tracking graph construction."""
import unittest

import torch
from scalabel.label.typing import Frame

from vis4d.struct import Images, InputSample, LabelInstances
from vis4d.unittest.utils import generate_dets

from .base import TrackGraphConfig
from .deepsort import DeepSORTTrackGraph


class TestDeepSortGraph(unittest.TestCase):
    """Test cases for deepsort tracking graph construction."""

    kalman_filter_paras = {
        "cov_P0": [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        "cov_motion_Q": [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        "cov_project_R": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ],
    }

    def test_track(self) -> None:
        """Testcase for tracking function."""
        tracker = DeepSORTTrackGraph(
            TrackGraphConfig(
                type="deepsort", kf_parameters=self.kalman_filter_paras
            )
        )

        h, w, num_dets = 128, 128, 64

        sample = InputSample(
            [Frame(name="myframe", frameIndex=0)],
            Images(torch.empty(1, 128, 128, 3), image_sizes=[(128, 128)]),
        )
        dets = LabelInstances([generate_dets(h, w, num_dets)])
        embeddings = [torch.rand(num_dets, 128)]

        # feed same detections & embeddings --> should be matched to self
        result_t0 = tracker(sample, dets, embeddings=embeddings)
        result_t1 = tracker(sample, dets, embeddings=embeddings)
        result_t2 = tracker(sample, dets, embeddings=embeddings)
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
                (res.boxes[:, -1] >= tracker.cfg.min_confidence).all()
            )
