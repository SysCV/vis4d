"""Dummy 3D motion model."""
import torch

from .base import BaseMotionModel, MotionModelConfig


class Dummy3DMotionModelConfig(MotionModelConfig):
    """Dummy 3D motion model config."""

    motion_momentum: float = 0.9


class Dummy3DMotionModel(BaseMotionModel):
    """Dummy 3D motion model."""

    def __init__(self, cfg, detections_3d):
        """Initialize a motion model tracker using initial bounding box.

        Args:
            cfg: motion tracker config.
            detections_3d: x, y, z, h, w, l, ry, depth confidence
        """
        self.cfg = Dummy3DMotionModelConfig(**cfg.dict())

        bbox_3d = detections_3d[: self.cfg.motion_dims]
        info = detections_3d[self.cfg.motion_dims :]

        self.obj_state = torch.cat([bbox_3d, bbox_3d.new_zeros(3)])
        self.history = bbox_3d.new_zeros(
            self.cfg.num_frames, self.cfg.motion_dims
        )
        self.prev_ref = bbox_3d.clone()
        self.info = info

    def update(self, detections_3d):
        """Update the state vector with observed bbox."""
        bbox_3d = detections_3d[: self.cfg.motion_dims]
        info = detections_3d[self.cfg.motion_dims :]

        self.cfg.time_since_update = 0
        self.cfg.hits += 1
        self.cfg.hit_streak += 1

        self.obj_state += self.cfg.motion_momentum * (
            torch.cat([bbox_3d, bbox_3d.new_zeros(3)]) - self.obj_state
        )
        self.prev_ref = bbox_3d
        self.info = info

    def predict(self, *args, **kwargs):
        """Advance the state vector and returns the predicted bounding box."""
        self.cfg.age += 1
        if self.cfg.time_since_update > 0:
            self.cfg.hit_streak = 0
        self.cfg.time_since_update += 1

        return self.obj_state

    def get_state(self):
        """Return the current bounding box estimate."""
        return self.obj_state

    def get_history(self):
        """Return the history of estimates."""
        return self.history
