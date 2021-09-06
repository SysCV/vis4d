"""Random Sampler."""
from collections import defaultdict
from typing import List, Tuple

import torch

from vist.struct import Boxes2D

from ..matchers.base import MatchResult
from ..utils import nonzero_tuple
from .base import BaseSampler, SamplerConfig, SamplingResult
from .utils import prepare_target


class RandomSampler(BaseSampler):
    """Random sampler class."""

    def __init__(self, cfg: SamplerConfig):
        """Init."""
        super().__init__()
        self.cfg = cfg
        self.bg_label = 0

    def sample(
        self,
        matching: List[MatchResult],
        boxes: List[Boxes2D],
        targets: List[Boxes2D],
    ) -> SamplingResult:
        """Sample boxes randomly."""
        result = defaultdict(list)
        for match, box, target in zip(matching, boxes, targets):
            pos_idx, neg_idx = self._sample_labels(match.assigned_labels)
            sampled_idcs = torch.cat([pos_idx, neg_idx], dim=0)

            result["sampled_boxes"] += [box[sampled_idcs]]
            result["sampled_targets"] += [
                prepare_target(sampled_idcs, target, match.assigned_gt_indices)
            ]
            result["sampled_labels"] += [match.assigned_labels[sampled_idcs]]
            result["sampled_indices"] += [sampled_idcs]
            result["sampled_target_indices"] += [
                match.assigned_gt_indices[sampled_idcs]
            ]

        return SamplingResult(**result)

    def _sample_labels(
        self, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample indices from given labels."""
        positive = nonzero_tuple((labels != -1) & (labels != self.bg_label))[0]
        negative = nonzero_tuple(labels == self.bg_label)[0]

        num_pos = int(
            self.cfg.batch_size_per_image * self.cfg.positive_fraction
        )
        # protect against not enough positive examples
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.cfg.batch_size_per_image - num_pos
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
