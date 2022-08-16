"""Interface for Vis4D bounding box samplers."""
import abc
from typing import List, NamedTuple

import torch
from torch import nn

from vis4d.struct import Boxes2D

from ..matchers.base import BaseMatcher, MatchResult


class SamplingResult(NamedTuple):
    """Match result class. Stores expected result tensors. TODO update doc

    sampled_boxes: List[Boxes2D] Sampled Boxes.
    sampled_targets: List[Boxes2D] Assigned target for each sampled box.
    sampled_labels: List[Tensor] of {0, -1, 1} = {neg, ignore, pos}.
    sampled_indices: List[Tensor] Index of input Boxes2D.
    sampled_label_indices: List[Tensor] Index of assigned target for each
    sampled box.
    """

    sampled_boxes: torch.Tensor
    sampled_target_boxes: torch.Tensor
    sampled_target_classes: torch.Tensor
    sampled_labels: torch.Tensor
    sampled_indices: torch.Tensor
    sampled_target_indices: torch.Tensor


class BaseSampler(nn.Module):
    """Sampler base class."""

    def __init__(self, batch_size: int, positive_fraction: float) -> None:
        """Init."""
        super().__init__()
        self.batch_size = batch_size
        self.positive_fraction = positive_fraction

    @abc.abstractmethod
    def forward(
        self,
        matching: MatchResult,
        boxes: torch.Tensor,
        targets: torch.Tensor,
    ) -> SamplingResult:
        """Sample bounding boxes according to their struct."""
        raise NotImplementedError


@torch.no_grad()  # type: ignore
def match_and_sample_proposals(  # TODO update
    matcher: BaseMatcher,
    sampler: BaseSampler,
    proposals: List[torch.Tensor],
    targets: List[torch.Tensor],
    proposal_append_gt: bool,
) -> List[SamplingResult]:
    """Match proposals to targets and subsample."""

    return result
