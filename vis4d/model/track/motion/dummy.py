"""Dummy 3D motion model."""
import numpy as np
import torch

from .base import BaseMotionModel, MotionModelConfig


class Dummy3DMotionModelConfig(MotionModelConfig):
    """Dummy 3D motion model config."""

    motion_momentum: float = 0.9


class Dummy3DMotionModel(BaseMotionModel):
    """Dummy 3D motion model."""

    def __init__(self, cfg, detections_3d):
        """
        Initialises a motion model tracker using initial bounding box.
        Args:
            cfg: motion tracker config.
            detections_3d: x, y, z, h, w, l, ry, depth uncertainty
        """
        self.cfg = Dummy3DMotionModelConfig(**cfg.dict())

        bbox3D = detections_3d[: self.cfg.motion_dims]
        info = detections_3d[self.cfg.motion_dims :]

        self.obj_state = torch.cat([bbox3D, bbox3D.new_zeros(3)])
        self.history = bbox3D.new_zeros(
            self.cfg.num_frames, self.cfg.motion_dims
        )
        self.prev_ref = bbox3D.clone()
        self.info = info

    def update(self, detections_3d):
        """
        Updates the state vector with observed bbox.
        """
        bbox3D = detections_3d[: self.cfg.motion_dims]
        info = detections_3d[self.cfg.motion_dims :]

        self.cfg.time_since_update = 0
        self.cfg.hits += 1
        self.cfg.hit_streak += 1

        self.obj_state += self.cfg.motion_momentum * (
            torch.cat([bbox3D, bbox3D.new_zeros(3)]) - self.obj_state
        )
        self.prev_ref = bbox3D
        self.info = info

    def predict(self, update_state: bool = True):
        """
        Advances the state vector and returns the predicted bounding box
        estimate.
        """
        self.cfg.age += 1
        if self.cfg.time_since_update > 0:
            self.cfg.hit_streak = 0
        self.cfg.time_since_update += 1

        return self.obj_state

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.obj_state

    def get_history(self):
        """
        Returns the history of estimates.
        """
        return self.history
