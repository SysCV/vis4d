"""RoI Pooling module base."""
import abc
from typing import List, Tuple

import torch
from pydantic import BaseModel

from openmt.core.registry import RegistryHolder
from openmt.structures import Boxes2D


class RoIPoolerConfig(BaseModel, extra="allow"):
    type: str
    resolution: Tuple[int, int]


class BaseRoIPooler(torch.nn.Module, metaclass=RegistryHolder):
    """Base class for RoI poolers"""

    @abc.abstractmethod
    def pool(
        self, features: List[torch.Tensor], boxes: List[Boxes2D]
    ) -> List[torch.Tensor]:
        """Pool region features corresponding to the input bounding boxes from
        the given feature maps."""


def build_roi_pooler(cfg: RoIPoolerConfig):
    """Build the component."""
    registry = RegistryHolder.get_registry(__package__)
    if cfg.type in registry:
        return registry[cfg.type](cfg)
    else:
        raise NotImplementedError(f"RoIPooler {cfg.type} not found.")
