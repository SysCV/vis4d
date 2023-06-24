"""Test cases for SimOTA matcher.

Modified from mmdetection (https://github.com/open-mmlab/mmdetection).
"""
import unittest

import torch

from vis4d.op.box.matchers.sim_ota import SimOTAMatcher


class TestSimOTA(unittest.TestCase):
    """Test cases for SimOTA matcher."""

    def test_match(self) -> None:
        """Testcase for match function."""
        matcher = SimOTAMatcher(
            center_radius=2.5, candidate_topk=1, iou_weight=3.0, cls_weight=1.0
        )
        match_result = matcher(
            pred_scores=torch.FloatTensor([[0.2], [0.8]]),
            priors=torch.Tensor([[30, 30, 8, 8], [4, 5, 6, 7]]),
            decoded_bboxes=torch.Tensor([[23, 23, 43, 43], [4, 5, 6, 7]]),
            gt_bboxes=torch.Tensor([[23, 23, 43, 43]]),
            gt_labels=torch.LongTensor([0]),
        )

        expected_gt_inds = torch.LongTensor([0, 0])
        expected_labels = torch.LongTensor([1, 0])
        self.assertTrue(
            torch.isclose(
                match_result.assigned_gt_indices, expected_gt_inds
            ).all()
        )
        self.assertTrue(
            torch.isclose(match_result.assigned_labels, expected_labels).all()
        )

    def test_match_with_no_valid_bboxes(self):
        """Testcase for match function with no valid bboxes."""
        matcher = SimOTAMatcher(
            center_radius=2.5, candidate_topk=1, iou_weight=3.0, cls_weight=1.0
        )
        match_result = matcher(
            pred_scores=torch.FloatTensor([[0.2], [0.8]]),
            priors=torch.Tensor([[30, 30, 8, 8], [55, 55, 8, 8]]),
            decoded_bboxes=torch.Tensor(
                [[123, 123, 143, 143], [114, 151, 161, 171]]
            ),
            gt_bboxes=torch.Tensor([[0, 0, 1, 1]]),
            gt_labels=torch.LongTensor([0]),
        )

        expected_gt_inds = torch.LongTensor([0, 0])
        self.assertTrue(
            torch.isclose(
                match_result.assigned_gt_indices, expected_gt_inds
            ).all()
        )

    def test_match_with_empty_gt(self):
        """Testcase for match function with empty gt."""
        matcher = SimOTAMatcher(
            center_radius=2.5, candidate_topk=1, iou_weight=3.0, cls_weight=1.0
        )
        match_result = matcher(
            pred_scores=torch.FloatTensor([[0.2], [0.8]]),
            priors=torch.Tensor([[0, 12, 23, 34], [4, 5, 6, 7]]),
            decoded_bboxes=torch.Tensor([[[30, 40, 50, 60]], [[4, 5, 6, 7]]]),
            gt_bboxes=torch.empty(0, 4),
            gt_labels=torch.empty(0),
        )

        expected_gt_inds = torch.LongTensor([0, 0])
        self.assertTrue(
            torch.isclose(
                match_result.assigned_gt_indices, expected_gt_inds
            ).all()
        )
