"""Test cases for random sampler."""
import unittest

import torch

from vis4d.struct import Boxes2D

from ..matchers.base import MatchResult
from .base import SamplerConfig
from .random import RandomSampler


class TestRandom(unittest.TestCase):
    """Test cases for random sampler."""

    def test_sample(self) -> None:
        """Testcase for sample function."""
        samples_per_img = 10
        pos_fract = 0.5
        num_samples = 10
        num_gts = 3

        sampler = RandomSampler(
            SamplerConfig(
                type="random",
                batch_size_per_image=samples_per_img,
                positive_fraction=pos_fract,
            )
        )
        matching = [
            MatchResult(
                assigned_gt_indices=torch.zeros(num_samples),
                assigned_gt_iou=torch.ones(num_samples),
                assigned_labels=torch.ones(num_samples),
            )
        ]
        boxes = [Boxes2D(torch.rand(num_samples, 5))]
        targets = [Boxes2D(torch.rand(num_gts, 5), torch.zeros(num_gts))]
        sampling_result = sampler.sample(matching, boxes, targets)
        sampled_boxes, sampled_targets = (
            sampling_result.sampled_boxes,
            sampling_result.sampled_targets,
        )
        self.assertEqual(
            len(sampled_boxes[0]), int(samples_per_img * pos_fract)
        )
        self.assertEqual(len(sampled_boxes[0]), len(sampled_targets[0]))

        for target in sampled_targets[0]:
            self.assertTrue(
                torch.isclose(targets[0][0].boxes, target.boxes).all()
            )

        boxes = [Boxes2D(torch.rand(num_samples, 5))]
        targets = [Boxes2D.empty()]
        sampling_result = sampler.sample(matching, boxes, targets)
        sampled_boxes, sampled_targets = (
            sampling_result.sampled_boxes,
            sampling_result.sampled_targets,
        )
        self.assertEqual(len(sampled_boxes[0]), len(sampled_targets[0]))
