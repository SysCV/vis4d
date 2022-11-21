"""Testcases for combined sampler."""
from __future__ import annotations

import unittest

import torch

from ..matchers.base import MatchResult
from .combined import CombinedSampler


class TestCombined(unittest.TestCase):
    """Test cases for combined sampler."""

    @staticmethod
    def _get_boxes_targets(
        num_gts: int, num_samples: int
    ) -> list[MatchResult]:
        """Generate match, box target."""
        state = torch.random.get_rng_state()
        torch.random.set_rng_state(torch.manual_seed(0).get_state())
        matching = MatchResult(
            assigned_gt_indices=torch.randint(0, num_gts, (num_samples,)),
            assigned_gt_iou=torch.rand(num_samples),
            assigned_labels=torch.randint(-1, 2, (num_samples,)),
        )
        torch.random.set_rng_state(state)
        return matching

    def test_sample(self) -> None:
        """Testcase for sample function."""
        samples_per_img = 256
        pos_fract = 0.5
        num_samples = 512
        num_gts = 3

        sampler = CombinedSampler(
            batch_size=samples_per_img,
            positive_fraction=pos_fract,
            pos_strategy="instance_balanced",
            neg_strategy="iou_balanced",
        )
        matching = self._get_boxes_targets(num_gts, num_samples)
        smp_box_inds, smp_tgt_inds, smp_lbls = sampler(matching)
        assert (
            len(smp_box_inds)
            == len(smp_tgt_inds)
            == len(smp_lbls)
            == samples_per_img
        )
        assert torch.logical_and(
            smp_tgt_inds >= 0, smp_tgt_inds < num_gts
        ).all()
        assert not torch.logical_and(
            smp_lbls != -1, torch.logical_and(smp_lbls != 1, smp_lbls != 0)
        ).any()

        sampler = CombinedSampler(
            batch_size=samples_per_img,
            positive_fraction=pos_fract,
            pos_strategy="instance_balanced",
            neg_strategy="iou_balanced",
            floor_thr=0.1,
            num_bins=1,
        )
        matching = self._get_boxes_targets(num_gts, num_samples)
        sampler(matching)

        matching = self._get_boxes_targets(num_gts, 128)
        sampler(matching)
