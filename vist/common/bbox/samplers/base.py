"""Interface for VisT bounding box samplers."""
import abc
from typing import List, NamedTuple

import torch
from pydantic import BaseModel, Field

from vist.common.registry import RegistryHolder
from vist.struct import Boxes2D

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

    sampled_boxes: List[Boxes2D]
    sampled_targets: List[Boxes2D]
    sampled_labels: List[torch.Tensor]
    sampled_indices: List[torch.Tensor]
    sampled_target_indices: List[torch.Tensor]


class SamplerConfig(BaseModel, extra="allow"):
    """Sampler base config."""

    # Field(...) necessary for linter
    # See https://github.com/samuelcolvin/pydantic/issues/1899
    type: str = Field(...)
    batch_size_per_image: int = Field(...)
    positive_fraction: float = Field(...)


class BaseSampler(metaclass=RegistryHolder):
    """Sampler base class."""

    @abc.abstractmethod
    def sample(
        self,
        matching: List[MatchResult],
        boxes: List[Boxes2D],
        targets: List[Boxes2D],
    ) -> SamplingResult:
        """Sample bounding boxes according to their struct."""
        raise NotImplementedError


def build_sampler(cfg: SamplerConfig) -> BaseSampler:
    """Build a bounding box sampler from config."""
    registry = RegistryHolder.get_registry(BaseSampler)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseSampler)
        return module
    raise NotImplementedError(f"Sampler {cfg.type} not found.")


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
    matching = matcher.match(proposals, targets)
    return sampler.sample(matching, proposals, targets)
