"""Test cases for Vis4D models."""
import unittest

import torch
from scalabel.label.typing import Frame

from vis4d.struct import Images, InputSample
from vis4d.unittest.utils import generate_input_sample

from .deepsort import DeepSortSimilarityHead


def get_input_sample(frame_index: int) -> InputSample:
    """
    Creates a dummy input frame with a given frame index
    """
    return InputSample(
        [Frame(name="myframe", frameIndex=frame_index)],
        Images(torch.empty(1, 128, 128, 3), image_sizes=[(128, 128)]),
    )


class TestDeepSortSimilarityHeader(unittest.TestCase):
    """Test cases for kalman filter."""

    n_features = 128
    n_detections = 10
    head = DeepSortSimilarityHead(fc_out_dim=n_features)

    def test_head(self) -> None:
        """Tests if the head can be called and results in a valid loss"""
        inputs = [
            generate_input_sample(
                256, 256, 1, self.n_detections, use_score=True, track_ids=True
            ),
            generate_input_sample(
                256, 256, 1, self.n_detections, use_score=True, track_ids=True
            ),
        ]
        loss, _ = self.head.forward_train(
            inputs=inputs,
            boxes=[det.targets.boxes2d for det in inputs],
            features=None,
            targets=[det.targets for det in inputs],
        )
        # Check loss is generated
        self.assertTrue(len(loss.keys()) != 0)

        out = self.head.forward_test(
            inputs=inputs[0], boxes=inputs[0].targets.boxes2d, features=None
        )
        # Check dimension matches
        self.assertTrue(
            list(out[0].shape) == [self.n_detections, self.n_features]
        )
