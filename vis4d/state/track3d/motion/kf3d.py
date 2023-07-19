"""Kalman Filter 3D motion model."""
from __future__ import annotations

from torch import Tensor

from vis4d.common.typing import ArgsType
from vis4d.op.track3d.motion.kf3d import (
    kf3d_init,
    kf3d_init_mean_cov,
    kf3d_predict,
    kf3d_update,
)

from .base import BaseMotionModel


class KF3DMotionModel(BaseMotionModel):
    """Kalman filter 3D motion model."""

    def __init__(
        self,
        *args: ArgsType,
        obs_3d: Tensor,
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__(*args, **kwargs)
        self.device = obs_3d.device

        # F, H, Q, R
        (
            self._motion_mat,
            self._update_mat,
            self._cov_motion_q,
            self._cov_project_r,
        ) = kf3d_init(self.motion_dims)

        self._motion_mat = self._motion_mat.to(self.device)
        self._update_mat = self._update_mat.to(self.device)
        self._cov_motion_q = self._cov_motion_q.to(self.device)
        self._cov_project_r = self._cov_project_r.to(self.device)

        self.mean, self.covariance = kf3d_init_mean_cov(
            obs_3d, self.motion_dims
        )

    def update(self, obs_3d: Tensor, info: Tensor) -> None:
        """Update the state."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        self.mean, self.covariance = kf3d_update(
            self._update_mat,
            self._cov_project_r,
            self.mean,
            self.covariance,
            obs_3d,
        )

    def predict_velocity(self) -> Tensor:
        """Predict velocity."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        pred_loc, _ = kf3d_predict(
            self._motion_mat,
            self._cov_motion_q,
            self.mean,
            self.covariance,
        )
        return (pred_loc[:3] - self.mean[:3]) * self.fps

    def predict(self, update_state: bool = True) -> Tensor:
        """Predict the state."""
        self.mean, self.covariance = kf3d_predict(
            self._motion_mat,
            self._cov_motion_q,
            self.mean,
            self.covariance,
        )

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.mean

    def get_state(self) -> Tensor:
        """Returns the current bounding box estimate."""
        return self.mean
