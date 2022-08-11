"""Interface for Vis4D bounding box samplers."""
import abc
from typing import List, NamedTuple

import torch
from torch import nn

from vis4d.struct import Boxes2D

from ..matchers.base import BaseMatcher, MatchResult


class SamplingResult(NamedTuple):
    """Match result class. Stores expected result tensors.

    sampled_boxes: List[Boxes2D] Sampled Boxes.
    sampled_targets: List[Boxes2D] Assigned target for each sampled box.
    sampled_labels: List[Tensor] of {0, -1, 1} = {neg, ignore, pos}.
    sampled_indices: List[Tensor] Index of input Boxes2D.
    sampled_label_indices: List[Tensor] Index of assigned target for each
    sampled box.
    """

    sampled_boxes: List[torch.Tensor]
    sampled_targets: List[torch.Tensor]
    sampled_labels: List[torch.Tensor]
    sampled_indices: List[torch.Tensor]
    sampled_target_indices: List[torch.Tensor]


class BaseSampler(nn.Module):
    """Sampler base class."""

    def __init__(
        self, batch_size_per_image: int, positive_fraction: float
    ) -> None:
        """Init."""
        super().__init__()
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    @abc.abstractmethod
    def forward(
        self,
        matching: List[MatchResult],
        boxes: List[Boxes2D],
        targets: List[Boxes2D],
    ) -> SamplingResult:
        """Sample bounding boxes according to their struct."""
        raise NotImplementedError


@torch.no_grad()  # type: ignore
def match_and_sample_proposals(
    matcher: BaseMatcher,
    sampler: BaseSampler,
    proposals: List[Boxes2D],
    targets: List[Boxes2D],
    proposal_append_gt: bool,
) -> SamplingResult:
    """Match proposals to targets and subsample."""
    if proposal_append_gt:
        proposals = [Boxes2D.merge([p, t]) for p, t in zip(proposals, targets)]
    matching = matcher(proposals, targets)
    return sampler(matching, proposals, targets)
