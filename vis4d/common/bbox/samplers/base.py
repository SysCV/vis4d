"""Interface for Vis4D bounding box samplers."""
import abc
from typing import List, NamedTuple, Tuple

import torch
from torch import Tensor, nn

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
        target_boxes: torch.Tensor,
        target_classes: torch.Tensor,
    ) -> SamplingResult:
        """Sample bounding boxes according to their struct."""
        raise NotImplementedError


@torch.no_grad()  # type: ignore
def match_and_sample_proposals(
    matcher: BaseMatcher,
    sampler: BaseSampler,
    proposals: List[torch.Tensor],
    scores: List[torch.Tensor],
    target_boxes: List[torch.Tensor],
    target_classes: List[torch.Tensor],
    proposal_append_gt: bool,
) -> Tuple[
    List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]
]:
    """Match proposals to targets and subsample."""

    with torch.no_grad():
        sampling_results = []
        for i, (p, s, tb, tc) in enumerate(
            zip(proposals, scores, target_boxes, target_classes)
        ):
            if proposal_append_gt:
                proposals[i] = torch.cat((p, tb), 0)
                scores[i] = torch.cat(
                    (
                        s,
                        s.new_ones(
                            (len(tb)),
                        ),
                    ),
                    0,
                )
            sampling_results.append(sampler(matcher(p, tb), p, tb, tc))

    proposals = [r.sampled_boxes for r in sampling_results]
    scores = [s[r.sampled_indices] for s, r in zip(scores, sampling_results)]
    sampled_target_boxes = [r.sampled_target_boxes for r in sampling_results]
    sampled_target_classes = [
        r.sampled_target_classes for r in sampling_results
    ]
    sampled_labels = [r.sampled_labels for r in sampling_results]

    return (
        proposals,
        scores,
        sampled_target_boxes,
        sampled_target_classes,
        sampled_labels,
    )
