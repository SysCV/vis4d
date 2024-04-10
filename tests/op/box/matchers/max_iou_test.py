"""Test cases for max iou matcher."""

import unittest

import torch

from tests.util import generate_boxes
from vis4d.op.box.matchers.max_iou import MaxIoUMatcher


class TestMaxIoUMatcher(unittest.TestCase):
    """Test cases for max iou matcher."""

    def test_match(self) -> None:
        """Testcase for match function."""
        num_boxes = 10

        boxes = generate_boxes(128, 128, num_boxes)[0]
        matcher = MaxIoUMatcher(
            thresholds=[0.3, 0.5],
            labels=[0, -1, 1],
            allow_low_quality_matches=True,
        )
        match_result = matcher(boxes[0], boxes[0])
        self.assertTrue(
            match_result.assigned_gt_indices.numpy().tolist()
            == list(range(num_boxes))
        )

        match_result = matcher(boxes[0], torch.empty([0, 4]))
        self.assertTrue((match_result.assigned_labels == 0.0).all())

        match_result = matcher(torch.empty([0, 4]), boxes[0])
        self.assertEqual(len(match_result.assigned_gt_indices), 0)
