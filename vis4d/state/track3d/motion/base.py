"""Motion model base class."""

from torch import Tensor


class BaseMotionModel:
    """Base class for motion model."""

    def __init__(
        self,
        num_frames: int,
        motion_dims: int,
        hits: int = 1,
        hit_streak: int = 0,
        time_since_update: int = 0,
        age: int = 0,
        fps: int = 1,
    ) -> None:
        """Creates an instance of the class."""
        self.num_frames = num_frames
        self.motion_dims = motion_dims
        self.hits = hits
        self.hit_streak = hit_streak
        self.time_since_update = time_since_update
        self.age = age
        self.fps = fps

    def update(self, obs_3d: Tensor, info: Tensor) -> None:
        """Update the state."""
        raise NotImplementedError()

    def predict_velocity(self) -> Tensor:
        """Predict velocity."""
        raise NotImplementedError()

    def predict(self, update_state: bool = True) -> Tensor:
        """Predict the state."""
        raise NotImplementedError()

    def get_state(self) -> Tensor:
        """Get the state."""
        raise NotImplementedError()


def update_array(origin_array: Tensor, input_array: Tensor) -> Tensor:
    """Update array according the input."""
    new_array = origin_array.clone()
    new_array[:-1] = origin_array[1:]
    new_array[-1:] = input_array
    return new_array
