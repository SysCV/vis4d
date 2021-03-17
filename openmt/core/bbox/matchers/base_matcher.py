"""Matchers."""
import abc
from typing import List, NamedTuple

import torch

from openmt.config import Matcher
from openmt.core.registry import RegistryHolder
from openmt.structures import Boxes2D


class MatchResult(NamedTuple):
    assigned_gt_indices: torch.Tensor  # Tensor of [0, M) where M = num gt
    assigned_labels: torch.Tensor  # Tensor of {0, -1, 1} = {neg, igonre, pos}


class BaseMatcher(metaclass=RegistryHolder):
    @abc.abstractmethod
    def match(
        self, boxes: List[Boxes2D], targets: List[Boxes2D]
    ) -> MatchResult:
        """Match bounding boxes according to their structures."""

        raise NotImplementedError


def build_matcher(cfg: Matcher) -> BaseMatcher:
    """Build a bounding box matcher from config."""
    model_registry = RegistryHolder.get_registry(__package__)
    if cfg.type in model_registry:
        return model_registry[cfg.type](cfg)
    else:
        raise NotImplementedError(f"Matcher {cfg.type} not found.")
