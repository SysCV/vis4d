"""Test cases for samplers."""
import unittest

import torch

from openmt.struct import Boxes2D

from ..matchers.base import MatchResult
from .base import SamplerConfig
from .combined import CombinedSampler
from .random import RandomSampler


class TestCombined(unittest.TestCase):
    """Test cases for combined sampler."""

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

        matching = [
            MatchResult(
                assigned_gt_indices=torch.randint(0, num_gts, (num_samples,)),
                assigned_gt_iou=torch.rand(num_samples),
                assigned_labels=torch.randint(-1, 2, (num_samples,)),
            )
        ]
        boxes = [Boxes2D(torch.rand(num_samples, 5))]
        targets = [Boxes2D(torch.rand(num_gts, 5))]
        sampled_boxes, sampled_targets = sampler.sample(
            matching, boxes, targets
        )
        sampled_boxes, sampled_targets = sampled_boxes[0], sampled_targets[0]
        self.assertEqual(len(sampled_boxes), samples_per_img)
        self.assertEqual(len(sampled_boxes), len(sampled_targets))

        sampled_idx = []
        for sampled_target in sampled_targets:
            found = False
            for i, target in enumerate(targets[0]):
                if torch.isclose(target.boxes, sampled_target.boxes).all():
                    sampled_idx.append(i)
                    found = True
            self.assertTrue(found)

        self.assertEqual(set(sampled_idx), set(range(num_gts)))


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
                type="random", batch_size_per_image=samples_per_img, positive_fraction=pos_fract
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
        targets = [Boxes2D(torch.rand(num_gts, 5))]
        sampled_boxes, sampled_targets = sampler.sample(
            matching, boxes, targets
        )
        sampled_boxes, sampled_targets = sampled_boxes[0], sampled_targets[0]
        self.assertEqual(len(sampled_boxes), int(samples_per_img * pos_fract))
        self.assertEqual(len(sampled_boxes), len(sampled_targets))

        for target in sampled_targets:
            self.assertTrue(
                torch.isclose(targets[0][0].boxes, target.boxes).all()
            )
