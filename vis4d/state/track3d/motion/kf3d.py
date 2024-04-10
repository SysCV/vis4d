"""Kalman Filter 3D motion model."""

from __future__ import annotations

import torch
from torch import Tensor

from vis4d.common.typing import ArgsType
from vis4d.op.geometry.rotation import acute_angle, normalize_angle
from vis4d.op.motion.kalman_filter import predict, update

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
        ) = self._kf3d_init()

        self._motion_mat = self._motion_mat.to(self.device)
        self._update_mat = self._update_mat.to(self.device)
        self._cov_motion_q = self._cov_motion_q.to(self.device)
        self._cov_project_r = self._cov_project_r.to(self.device)

        self.mean, self.covariance = self._init_mean_cov(obs_3d)

    def _kf3d_init(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """KF3D init function."""
        motion_mat = torch.Tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        update_mat = torch.Tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )

        cov_motion_q = torch.eye(self.motion_dims + 3)
        cov_motion_q[self.motion_dims :, self.motion_dims :] *= 0.01

        cov_project_r = torch.eye(self.motion_dims)
        return motion_mat, update_mat, cov_motion_q, cov_project_r

    def _init_mean_cov(self, obs_3d: Tensor) -> tuple[Tensor, Tensor]:
        """Init KF3D mean and covariance."""
        mean = torch.zeros(self.motion_dims + 3).to(obs_3d.device)
        mean[: self.motion_dims] = obs_3d
        covariance = torch.eye(self.motion_dims + 3).to(obs_3d.device) * 10.0
        covariance[self.motion_dims :, self.motion_dims :] *= 1000.0
        return mean, covariance

    def update(self, obs_3d: Tensor, info: Tensor) -> None:
        """Update the state."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        self.mean[6] = normalize_angle(self.mean[6])
        obs_3d[6] = normalize_angle(obs_3d[6])

        self.mean[6] = acute_angle(self.mean[6], obs_3d[6])

        self.mean, self.covariance = update(
            self._update_mat,
            self._cov_project_r,
            self.mean,
            self.covariance,
            obs_3d,
        )
        self.mean[6] = normalize_angle(self.mean[6])

    def predict_velocity(self) -> Tensor:
        """Predict velocity."""
        pred_loc, _ = predict(
            self._motion_mat,
            self._cov_motion_q,
            self.mean,
            self.covariance,
        )
        return (pred_loc[:3] - self.mean[:3]) * self.fps

    def predict(self, update_state: bool = True) -> Tensor:
        """Predict the state."""
        self.mean, self.covariance = predict(
            self._motion_mat, self._cov_motion_q, self.mean, self.covariance
        )

        self.mean[6] = normalize_angle(self.mean[6])

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.mean

    def get_state(self) -> Tensor:
        """Returns the current bounding box estimate."""
        return self.mean
