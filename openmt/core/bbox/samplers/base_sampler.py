"""Interface for openMT bounding box samplers."""

import abc
from typing import List, Tuple

from openmt.config import Sampler
from openmt.core.registry import RegistryHolder
from openmt.structures import Boxes2D

from ..matchers.base_matcher import MatchResult


class BaseSampler(metaclass=RegistryHolder):
    @abc.abstractmethod
    def sample(
        self,
        matching: MatchResult,
        boxes: List[Boxes2D],
        targets: List[Boxes2D],
    ) -> Tuple[List[Boxes2D], List[Boxes2D]]:
        """Sample bounding boxes according to their structures."""
        raise NotImplementedError


def build_sampler(cfg: Sampler) -> BaseSampler:
    """Build a bounding box sampler from config."""
    model_registry = RegistryHolder.get_registry(__package__)
    if cfg.type in model_registry:
        return model_registry[cfg.type](cfg)
    else:
        raise NotImplementedError(f"Sampler {cfg.type} not found.")
