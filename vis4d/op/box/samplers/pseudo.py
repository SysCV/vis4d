"""Random Sampler."""
from typing import Tuple

import torch

from ..matchers.base import MatchResult
from .base import BaseSampler, SamplingResult


class PseudoSampler(BaseSampler):
    """Pseudo sampler class."""

    def forward(self, matching: MatchResult) -> SamplingResult:
        """Sample boxes randomly."""
        pos_idx, neg_idx = self._sample_labels(matching.assigned_labels)
        sampled_idcs = torch.cat([pos_idx, neg_idx], dim=0)
        return SamplingResult(
            sampled_box_indices=sampled_idcs,
            sampled_target_indices=matching.assigned_gt_indices[sampled_idcs],
            sampled_labels=matching.assigned_labels[sampled_idcs],
        )

    @staticmethod
    def _sample_labels(
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample indices from given labels."""
        positive = ((labels != -1) & (labels != 0)).nonzero()[:, 0]
        negative = (labels == 0).nonzero()[:, 0]
        return positive, negative