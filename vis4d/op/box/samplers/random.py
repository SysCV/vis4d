"""Random Sampler."""

from __future__ import annotations

import torch

from vis4d.common import ArgsType

from ..matchers.base import MatchResult
from .base import Sampler, SamplingResult


class RandomSampler(Sampler):
    """Random sampler class."""

    def __init__(
        self,
        *args: ArgsType,
        bg_label: int = 0,
        **kwargs: ArgsType,
    ):
        """Creates an instance of the class."""
        super().__init__(*args, **kwargs)
        self.bg_label = bg_label

    def forward(
        self,
        matching: MatchResult,
    ) -> SamplingResult:
        """Sample boxes randomly."""
        pos_idx, neg_idx = self._sample_labels(matching.assigned_labels)
        sampled_idcs = torch.cat([pos_idx, neg_idx], dim=0)
        return SamplingResult(
            sampled_box_indices=sampled_idcs,
            sampled_target_indices=matching.assigned_gt_indices[sampled_idcs],
            sampled_labels=matching.assigned_labels[sampled_idcs],
        )

    def _sample_labels(
        self, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample indices from given labels."""
        positive = ((labels != -1) & (labels != self.bg_label)).nonzero()[:, 0]
        negative = torch.eq(labels, self.bg_label).nonzero()[:, 0]

        num_pos = int(self.batch_size * self.positive_fraction)
        # protect against not enough positive examples
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.batch_size - num_pos
        # protect against not enough negative examples
        num_neg = min(negative.numel(), num_neg)

        # randomly select positive and negative examples
        perm1 = torch.randperm(positive.numel(), device=positive.device)[
            :num_pos
        ]
        perm2 = torch.randperm(negative.numel(), device=negative.device)[
            :num_neg
        ]

        pos_idx = positive[perm1]
        neg_idx = negative[perm2]
        return pos_idx, neg_idx
