"""Kalman Filter motion model."""
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from vis4d.op.geometry.rotation import normalize_angle, acute_angle
from vis4d.common import ArgsType

from .base import BaseMotionModel

from filterpy.kalman import KalmanFilter, predict


class KF3DMotionModel(BaseMotionModel):
    """Kalman Filter 3D motion model."""

    def __init__(
        self,
        obs_3d: torch.Tensor,
        *args: ArgsType,
        init_flag: bool = True,
        **kwargs: ArgsType,
    ) -> None:
        """Initialize a motion model using initial bounding box."""
        super().__init__(*args, **kwargs)
        self.init_flag = init_flag
        self.device = obs_3d.device

        self.kf = KalmanFilter(
            dim_x=self.motion_dims + 3, dim_z=self.motion_dims
        )
        # state transition matrix
        self.kf.F = np.array(
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
        # measurement function
        self.kf.H = np.array(
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
        # state uncertainty, give high uncertainty to
        self.kf.P[self.motion_dims :, self.motion_dims :] *= 1000.0
        # the unobservable initial velocities, covariance matrix
        self.kf.P *= 10.0

        # process uncertainty
        self.kf.Q[self.motion_dims :, self.motion_dims :] *= 0.01

        bbox_3d = obs_3d.cpu().numpy()

        self.kf.x[: self.motion_dims] = bbox_3d.reshape((self.motion_dims, 1))
        self.prev_ref = bbox_3d

    def update(self, obs_3d: torch.Tensor) -> None:  # type: ignore
        """Updates the state vector with observed bbox."""
        bbox_3d = obs_3d.cpu().numpy()

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        self.kf.x[6] = normalize_angle(self.kf.x[6])
        bbox_3d[6] = normalize_angle(bbox_3d[6])

        self.kf.x[6] = acute_angle(self.kf.x[6], bbox_3d[6])

        # Update the bbox3D
        self.kf.update(bbox_3d)

        self.kf.x[6] = normalize_angle(self.kf.x[6])

        self.prev_ref = self.kf.x[: self.motion_dims].flatten()

    def predict_velocity(self) -> torch.Tensor:  # type: ignore
        """Predict velocity."""
        pred_loc, _ = predict(self.kf.x, self.kf.P, self.kf.F, self.kf.Q)

        return torch.tensor(
            pred_loc.flatten()[:3] - self.prev_ref[:3], device=self.device
        )

    def predict(self, *args: ArgsType, **kwargs: ArgsType) -> torch.Tensor:
        """Advances the state vector and returns the predicted bounding box."""
        self.kf.predict()

        self.kf.x[6] = normalize_angle(self.kf.x[6])

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return torch.tensor(self.kf.x.flatten(), device=self.device)

    def get_state(self):
        """Returns the current bounding box estimate."""
        return torch.tensor(self.kf.x.flatten(), device=self.device)
