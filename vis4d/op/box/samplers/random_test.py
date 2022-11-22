"""Test cases for random sampler."""
import unittest

import torch

from ..matchers.base import MatchResult
from .random import RandomSampler


class TestRandom(unittest.TestCase):
    """Test cases for random sampler."""

    def test_sample(self) -> None:
        """Testcase for sample function."""
        samples_per_img = 10
        pos_fract = 0.5
        num_samples = 10

        sampler = RandomSampler(
            batch_size=samples_per_img, positive_fraction=pos_fract
        )
        matching = MatchResult(
            assigned_gt_indices=torch.zeros(num_samples),
            assigned_gt_iou=torch.ones(num_samples),
            assigned_labels=torch.ones(num_samples),
        )
        smp_box_inds, smp_tgt_inds, smp_lbls = sampler(matching)

        assert len(smp_box_inds) == int(samples_per_img * pos_fract)
        assert len(smp_box_inds) == len(smp_tgt_inds) == len(smp_lbls)
