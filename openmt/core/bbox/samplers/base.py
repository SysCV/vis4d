"""Interface for openMT bounding box samplers."""

import abc
from typing import List, Tuple

from pydantic import BaseModel, Field

from openmt.core.registry import RegistryHolder
from openmt.struct import Boxes2D

from ..matchers.base import MatchResult


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
        matching: MatchResult,
        boxes: List[Boxes2D],
        targets: List[Boxes2D],
    ) -> Tuple[List[Boxes2D], List[Boxes2D]]:
        """Sample bounding boxes according to their struct."""
        raise NotImplementedError


def build_sampler(cfg: SamplerConfig) -> BaseSampler:
    """Build a bounding box sampler from config."""
    registry = RegistryHolder.get_registry(__package__)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseSampler)
        return module
    raise NotImplementedError(f"Sampler {cfg.type} not found.")
