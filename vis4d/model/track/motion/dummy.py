"""Dummy 3D motion model."""
import torch

from vis4d.struct import ArgsType

from .base import BaseMotionModel


class Dummy3DMotionModel(BaseMotionModel):
    """Dummy 3D motion model."""

    def __init__(
        self,
        detections_3d: torch.Tensor,
        *args: ArgsType,
        motion_momentum: float = 0.9,
        **kwargs: ArgsType,
    ) -> None:
        """Initialize a motion model using initial bounding box."""
        super().__init__(*args, **kwargs)
        self.motion_momentum = motion_momentum

        bbox_3d = detections_3d[: self.motion_dims]
        info = detections_3d[self.motion_dims :]

        self.obj_state = torch.cat([bbox_3d, bbox_3d.new_zeros(3)])
        self.history = bbox_3d.new_zeros(self.num_frames, self.motion_dims)
        self.prev_ref = bbox_3d.clone()
        self.info = info

    def update(self, detections_3d: torch.Tensor) -> None:  # type: ignore
        """Update the state vector with observed bbox."""
        bbox_3d = detections_3d[: self.motion_dims]
        info = detections_3d[self.motion_dims :]

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        self.obj_state += self.motion_momentum * (
            torch.cat([bbox_3d, bbox_3d.new_zeros(3)]) - self.obj_state
        )
        self.prev_ref = bbox_3d
        self.info = info

    def predict(self) -> torch.Tensor:  # type: ignore
        """Advance the state vector and returns the predicted bounding box."""
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.obj_state

    def get_state(self) -> torch.Tensor:  # type: ignore
        """Return the current bounding box estimate."""
        return self.obj_state

    def get_history(self) -> torch.Tensor:  # type: ignore
        """Return the history of estimates."""
        return self.history
