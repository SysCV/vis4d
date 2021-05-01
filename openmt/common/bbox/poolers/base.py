"""RoI Pooling module base."""
import abc
from typing import List, Tuple

import torch
from pydantic import BaseModel, Field

from openmt.common.registry import RegistryHolder
from openmt.struct import Boxes2D


class RoIPoolerConfig(BaseModel, extra="allow"):
    """Base RoI pooler config."""

    type: str = Field(...)
    resolution: Tuple[int, int] = Field(...)


class BaseRoIPooler(metaclass=RegistryHolder):
    """Base class for RoI poolers."""

    @abc.abstractmethod
    def pool(
        self, features: List[torch.Tensor], boxes: List[Boxes2D]
    ) -> torch.Tensor:
        """Pool features in input bounding boxes from given feature maps."""
        raise NotImplementedError


def build_roi_pooler(cfg: RoIPoolerConfig) -> BaseRoIPooler:
    """Build an RoI pooler from config."""
    registry = RegistryHolder.get_registry(__package__)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseRoIPooler)
        return module
    raise NotImplementedError(f"RoIPooler {cfg.type} not found.")
