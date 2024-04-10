"""Interface for Vis4D bounding box samplers."""

from __future__ import annotations

import abc
from typing import NamedTuple

import torch
from torch import Tensor, nn

from ..matchers import Matcher, MatchResult


class SamplingResult(NamedTuple):
    """Sampling result class. Stores expected result tensors.

    sampled_box_indices (Tensor): Index of sampled boxes from input.
    sampled_target_indices (Tensor): Index of assigned target for each
        positive sampled box.
    sampled_labels (Tensor): {0, -1, 1} = {neg, ignore, pos}.
    """

    sampled_box_indices: Tensor
    sampled_target_indices: Tensor
    sampled_labels: Tensor


class Sampler(nn.Module):
    """Sampler base class."""

    def __init__(self, batch_size: int, positive_fraction: float) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.batch_size = batch_size
        self.positive_fraction = positive_fraction

    @abc.abstractmethod
    def forward(self, matching: MatchResult) -> SamplingResult:
        """Sample bounding boxes according to their struct."""
        raise NotImplementedError

    def __call__(self, matching: MatchResult) -> SamplingResult:
        """Type declaration."""
        return self._call_impl(matching)


def match_and_sample_proposals(
    matcher: Matcher,
    sampler: Sampler,
    proposal_boxes: list[Tensor],
    target_boxes: list[Tensor],
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Match proposals to targets and subsample.

    First, match the proposals to targets (ground truth labels) using the
    matcher. It is usually IoU matcher. The matching labels the proposals with
    positive or negative to show whether they are matched to an object.
    Second, the sampler will choose proposals based on certain criteria such as
    total proposal number and ratio of postives and negatives.
    """
    with torch.no_grad():
        matchings = tuple(
            matcher(prop_box, tgt_box)
            for prop_box, tgt_box in zip(proposal_boxes, target_boxes)
        )
        sampling_results = tuple(sampler(matchs) for matchs in matchings)
        return (
            [s.sampled_box_indices for s in sampling_results],
            [s.sampled_target_indices for s in sampling_results],
            [s.sampled_labels for s in sampling_results],
        )
