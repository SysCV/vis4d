"""Motion model base class."""
import abc

import torch
from torch import nn


class BaseMotionModel(nn.Module):
    """Base class for motion model."""

    def __init__(
        self,
        num_frames: int,
        motion_dims: int,
        hits: int = 1,
        hit_streak: int = 0,
        time_since_update: int = 0,
        age: int = 0,
    ) -> None:
        """Init."""
        super().__init__()
        self.num_frames = num_frames
        self.motion_dims = motion_dims
        self.hits = hits
        self.hit_streak = hit_streak
        self.time_since_update = time_since_update
        self.age = age

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
