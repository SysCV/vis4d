"""BBox coder base class."""
import abc
from typing import List

import torch
from pydantic import BaseModel, Field

from vist.common.registry import RegistryHolder
from vist.struct import Boxes2D, Intrinsics


class BaseBoxCoderConfig(BaseModel, extra="allow"):
    """Coder base config."""

    type: str = Field(...)


class BaseBoxCoder2D(metaclass=RegistryHolder):
    """Base class for box coders."""

    @abc.abstractmethod
    def encode(self, boxes: List[Boxes2D], targets: List[Boxes2D]) -> torch.Tensor:
        """Encode deltas between boxes and targets."""

    @abc.abstractmethod
    def decode(self, anchors: List[Boxes2D], box_deltas: torch.Tensor) -> Boxes2D:
        """Decode the predicted bboxes according to prediction and base
        boxes."""


def build_box2d_coder(cfg: BaseBoxCoderConfig) -> BaseBoxCoder2D:
    """Build a bounding box matcher from config."""
    registry = RegistryHolder.get_registry(BaseBoxCoder2D)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseBoxCoder2D)
        return module
    raise NotImplementedError(f"BoxCoder2D {cfg.type} not found.")
