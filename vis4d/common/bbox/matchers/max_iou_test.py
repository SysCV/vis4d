"""Test cases for max iou matcher."""
import unittest

import torch

from vis4d.struct import Boxes2D
from vis4d.unittest.utils import generate_dets

from .max_iou import MaxIoUMatcher


class TestRandom(unittest.TestCase):
    """Test cases for max iou matcher."""

    def test_match(self) -> None:
        """Testcase for sample function."""
        num_boxes = 10

        boxes = [generate_dets(128, 128, num_boxes)]
        matcher = MaxIoUMatcher(
            thresholds=[0.3, 0.5],
            labels=[0, -1, 1],
            allow_low_quality_matches=True,
        )
        match_result = matcher(boxes, boxes)[0]
        self.assertTrue(
            match_result.assigned_gt_indices.numpy().tolist()
            == list(range(num_boxes))
        )

        match_result = matcher(boxes, [Boxes2D(torch.empty(0, 5))])[0]
        self.assertTrue((match_result.assigned_labels == 0.0).all())

        match_result = matcher([Boxes2D(torch.empty(0, 5))], boxes)[0]
        self.assertEqual(len(match_result.assigned_gt_indices), 0)
