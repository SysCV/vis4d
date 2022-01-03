"""Random Sampler."""
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import torch

from vis4d.struct import Boxes2D

from ..matchers.base import MatchResult
from .base import BaseSampler, SamplingResult
from .utils import add_to_result


class RandomSampler(BaseSampler):
    """Random sampler class."""

    def __init__(
        self,
        batch_size_per_image: int,
        positive_fraction: float,
        bg_label: int = 0,
    ):
        """Init."""
        super().__init__(batch_size_per_image, positive_fraction)
        self.bg_label = bg_label

    def __call__(  # type: ignore
        self,
        matching: List[MatchResult],
        boxes: List[Boxes2D],
        targets: List[Boxes2D],
    ) -> SamplingResult:
        """Sample boxes randomly."""
        result: Dict[
            str, Union[List[Boxes2D], List[torch.Tensor]]
        ] = defaultdict(list)
        for match, box, target in zip(matching, boxes, targets):
            pos_idx, neg_idx = self._sample_labels(match.assigned_labels)
            sampled_idcs = torch.cat([pos_idx, neg_idx], dim=0)
            add_to_result(result, sampled_idcs, box, target, match)

        return SamplingResult(**result)

    def _sample_labels(
        self, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample indices from given labels."""
        positive = ((labels != -1) & (labels != self.bg_label)).nonzero()[:, 0]
        negative = (labels == self.bg_label).nonzero()[:, 0]

        num_pos = int(self.batch_size_per_image * self.positive_fraction)
        # protect against not enough positive examples
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.batch_size_per_image - num_pos
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
