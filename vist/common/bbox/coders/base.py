"""BBox coder base classes."""
import abc
from typing import List

import torch
from pydantic import BaseModel, Field

from vist.common.registry import RegistryHolder
from vist.struct import Boxes2D, Intrinsics, Boxes3D


class BaseBoxCoderConfig(BaseModel, extra="allow"):
    """Coder base config."""

    type: str = Field(...)


class BaseBoxCoder2D(metaclass=RegistryHolder):
    """Base class for 2D box coders."""

    @abc.abstractmethod
    def encode(self, boxes: List[Boxes2D], targets: List[Boxes2D]) -> List[torch.Tensor]:
        """Encode deltas between boxes and targets."""
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, boxes: List[Boxes2D], box_deltas: List[torch.Tensor]) -> List[Boxes2D]:
        """Decode the predicted box_deltas according to given base boxes."""
        raise NotImplementedError


class BaseBoxCoder3D(metaclass=RegistryHolder):
    """Base class for 3D box coders."""

    @abc.abstractmethod
    def encode(self, boxes: List[Boxes2D], targets: List[Boxes3D], intrinsics: Intrinsics) -> List[torch.Tensor]:
        """Encode deltas between boxes and targets given intrinsics."""
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, boxes: List[Boxes2D], box_deltas: List[torch.Tensor], intrinsics: Intrinsics) -> List[Boxes3D]:
        """Decode the predicted box_deltas according to given base boxes."""
        raise NotImplementedError


def build_box2d_coder(cfg: BaseBoxCoderConfig) -> BaseBoxCoder2D:
    """Build a 2D bounding box coder from config."""
    registry = RegistryHolder.get_registry(BaseBoxCoder2D)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseBoxCoder2D)
        return module
    raise NotImplementedError(f"BoxCoder2D {cfg.type} not found.")


def build_box3d_coder(cfg: BaseBoxCoderConfig) -> BaseBoxCoder3D:
    """Build a 3D bounding box coder from config."""
    registry = RegistryHolder.get_registry(BaseBoxCoder3D)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseBoxCoder3D)
        return module
    raise NotImplementedError(f"BoxCoder3D {cfg.type} not found.")
