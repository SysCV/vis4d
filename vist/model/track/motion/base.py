"""Motion model base class."""
import abc

import torch

from pydantic import BaseModel, Field
from typing import List, Optional

from vist.common.registry import RegistryHolder


class MotionModelConfig(BaseModel, extra="allow"):
    """Base config for motion tracker."""

    type: str = Field(...)
    num_frames: int
    motion_dims: int
    hits: int = 1
    hit_streak: int = 0
    time_since_update: int = 0
    age: int = 0


class BaseMotionModel(metaclass=RegistryHolder):  # type: ignore # pylint: disable=line-too-long
    """Base class for motion tracker."""

    @staticmethod
    def update_array(
        origin_array: torch.Tensor, input_array: torch.Tensor
    ) -> torch.Tensor:
        """Update array according the input."""
        new_array = origin_array.clone()
        new_array[:-1] = origin_array[1:]
        new_array[-1:] = input_array
        return new_array

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None:  # type: ignore
        """Update object state with observation."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> torch.Tensor:  # type: ignore
        """Advances the object state and return predicted bbox."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_state(self, *args, **kwargs) -> torch.Tensor:  # type: ignore
        """Returns the current bbox estimation."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_history(self, *args, **kwargs) -> torch.Tensor:  # type: ignore
        """Returns the history of estimation."""
        raise NotImplementedError


def build_motion_model(
    cfg: MotionModelConfig, detections: torch.Tensor
) -> BaseMotionModel:
    """Build a tracking graph optimize from config."""
    registry = RegistryHolder.get_registry(BaseMotionModel)
    if cfg.type in registry:
        module = registry[cfg.type](cfg, detections)
        assert isinstance(module, BaseMotionModel)
        return module
    raise NotImplementedError(f"Motion model {cfg.type} not found.")
