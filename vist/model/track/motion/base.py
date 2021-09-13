"""Motion model base class."""

import abc

import torch
from pydantic import BaseModel, Field
from typing import List, Optional


from vist.common.registry import RegistryHolder
from vist.struct import Boxes2D, Boxes3D


class MotionTrackerConfig(BaseModel, extra="allow"):
    """Base config for motion tracker."""

    type: str = Field(...)
    nfr: int
    loc_dim: int


class BaseMotionTracker(metaclass=RegistryHolder):  # type: ignore # pylint: disable=line-too-long
    """Base class for motion model."""

    count = 0

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None:  # type: ignore
        """Update object state with observation."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> Boxes2D:  # type: ignore
        """Advances the object state and return predicted bbox."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_state(self) -> Boxes2D:  # type: ignore
        """Returns the current bbox estimation."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_history(self) -> List[Boxes2D]:  # type: ignore
        """Returns the history of estimation."""
        raise NotImplementedError


def build_motion_tracker(
    device: str,
    cfg: MotionTrackerConfig,
    detections,
) -> BaseMotionTracker:
    """Build a tracking graph optimize from config."""
    registry = RegistryHolder.get_registry(BaseMotionTracker)
    if cfg.type in registry:
        module = registry[cfg.type](device, cfg, detections)
        assert isinstance(module, BaseMotionTracker)
        return module
    raise NotImplementedError(f"Motion model {cfg.type} not found.")
