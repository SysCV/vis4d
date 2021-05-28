"""Matchers."""
import abc
from typing import List, NamedTuple

import torch
from pydantic import BaseModel, Field

from openmt.common.registry import RegistryHolder
from openmt.struct import Boxes2D


class MatcherConfig(BaseModel, extra="allow"):
    """Matcher base config."""

    type: str = Field(...)


class MatchResult(NamedTuple):
    """Match result class. Stores expected result tensors.

    assigned_gt_indices: torch.Tensor - Tensor of [0, M) where M = num gt
    assigned_gt_iou: torch.Tensor  - Tensor with IoU to assigned GT
    assigned_labels: torch.Tensor  - Tensor of {0, -1, 1} = {neg, ignore, pos}
    """

    assigned_gt_indices: torch.Tensor
    assigned_gt_iou: torch.Tensor
    assigned_labels: torch.Tensor


class BaseMatcher(metaclass=RegistryHolder):
    """Base class for box / target matchers."""

    @abc.abstractmethod
    def match(
        self, boxes: List[Boxes2D], targets: List[Boxes2D]
    ) -> List[MatchResult]:
        """Match bounding boxes according to their struct."""
        raise NotImplementedError


def build_matcher(cfg: MatcherConfig) -> BaseMatcher:
    """Build a bounding box matcher from config."""
    registry = RegistryHolder.get_registry(__package__)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseMatcher)
        return module
    raise NotImplementedError(f"Matcher {cfg.type} not found.")
