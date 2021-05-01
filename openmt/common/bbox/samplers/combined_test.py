"""Testcases for combined sampler."""
import unittest
from typing import List, Tuple

import torch

from openmt.struct import Boxes2D

from ..matchers.base import MatchResult
from .base import SamplerConfig
from .combined import CombinedSampler


class TestCombined(unittest.TestCase):
    """Test cases for combined sampler."""

    @staticmethod
    def _get_boxes_targets(
        num_gts: int, num_samples: int
    ) -> Tuple[List[MatchResult], List[Boxes2D], List[Boxes2D]]:
        """Generate match, box target."""
        state = torch.random.get_rng_state()
        torch.random.set_rng_state(torch.manual_seed(0).get_state())
        matching = [
            MatchResult(
                assigned_gt_indices=torch.randint(0, num_gts, (num_samples,)),
                assigned_gt_iou=torch.rand(num_samples),
                assigned_labels=torch.randint(-1, 2, (num_samples,)),
            )
        ]
        boxes = [Boxes2D(torch.rand(num_samples, 5))]
        targets = [Boxes2D(torch.rand(num_gts, 5), torch.zeros(num_gts))]
        torch.random.set_rng_state(state)
        return matching, boxes, targets

    def test_sample(self) -> None:
        """Testcase for sample function."""
        samples_per_img = 256
        pos_fract = 0.5
        num_samples = 512
        num_gts = 3

        sampler = CombinedSampler(
            SamplerConfig(
                type="combined",
                batch_size_per_image=samples_per_img,
                positive_fraction=pos_fract,
                pos_strategy="instance_balanced",
                neg_strategy="iou_balanced",
            )
        )
        matching, boxes, targets = self._get_boxes_targets(
            num_gts, num_samples
        )
        sampled_boxes, sampled_targets = sampler.sample(
            matching, boxes, targets
        )
        self.assertEqual(len(sampled_boxes[0]), samples_per_img)
        self.assertEqual(len(sampled_boxes[0]), len(sampled_targets[0]))

        sampled_idx = []
        for sampled_target in sampled_targets[0]:  # type: ignore
            found = False
            for i, target in enumerate(targets[0]):  # type: ignore
                if torch.isclose(target.boxes, sampled_target.boxes).all():
                    sampled_idx.append(i)
                    found = True
            self.assertTrue(found)

        self.assertEqual(set(sampled_idx), set(range(num_gts)))

        sampler = CombinedSampler(
            SamplerConfig(
                type="combined",
                batch_size_per_image=samples_per_img,
                positive_fraction=pos_fract,
                pos_strategy="instance_balanced",
                neg_strategy="iou_balanced",
                floor_thr=0.1,
                num_bins=1,
            )
        )
        matching, boxes, targets = self._get_boxes_targets(
            num_gts, num_samples
        )
        sampler.sample(matching, boxes, targets)

        matching, boxes, targets = self._get_boxes_targets(num_gts, 128)
        sampler.sample(matching, boxes, targets)
