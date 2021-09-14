"""Kalman Filter 3D motion model."""
import numpy as np
import torch

from filterpy.kalman import KalmanFilter

from .base import BaseMotionModel, MotionModelConfig


class KF3DMotionModelConfig(MotionModelConfig):
    """Kalman Filter 3D motion model config."""

    dim_x: int = 10
    init_p_uncertainty: float = 1000.0
    init_q_uncertainty: float = 0.01
    init_velocity: float = 10.0


class KF3DMotionModel(BaseMotionModel):
    """LSTM 3D motion model tracker."""

    def __init__(self, cfg, detections_3d):
        """
        Initialises a tracker using initial bounding box.

        Args:
            cfg: motion tracker config.
            detections_3d: x, y, z, h, w, l, ry, depth uncertainty
        """
        self.cfg = KF3DMotionModelConfig(**cfg.dict())
        self.device = detections_3d.device

        bbox3D = detections_3d[: self.cfg.motion_dims].detach().cpu().numpy()
        info = detections_3d[self.cfg.motion_dims :].detach().cpu().numpy()

        # define constant velocity model
        self.kf = KalmanFilter(
            dim_x=self.cfg.dim_x, dim_z=self.cfg.motion_dims
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
        # measurement function,
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
        self.kf.P[
            self.cfg.motion_dims :, self.cfg.motion_dims :
        ] *= self.cfg.init_p_uncertainty
        # the unobservable initial velocities, covariance matrix
        self.kf.P *= self.cfg.init_velocity

        # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
        self.kf.Q[
            self.cfg.motion_dims :, self.cfg.motion_dims :
        ] *= self.cfg.init_q_uncertainty

        self.kf.x[: self.cfg.motion_dims] = bbox3D.reshape(
            (self.cfg.motion_dims, 1)
        )

        self.history = []
        self.prev_ref = bbox3D

    def _update_history(self, bbox3D):
        self.history = self.history[1:] + [bbox3D - self.prev_ref]

    def _init_history(self, bbox3D):
        self.history = [bbox3D - self.prev_ref] * self.nfr

    def update(self, detections_3d):
        """
        Updates the state vector with observed bbox.
        """
        bbox3D = detections_3d[: self.cfg.motion_dims].detach().cpu().numpy()
        info = detections_3d[self.cfg.motion_dims :].detach().cpu().numpy()

        self.cfg.time_since_update = 0
        self.cfg.hits += 1
        self.cfg.hit_streak += 1

        # orientation correction
        if self.kf.x[6] >= np.pi:
            self.kf.x[6] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[6] < -np.pi:
            self.kf.x[6] += np.pi * 2

        new_theta = bbox3D[3]
        if new_theta >= np.pi:
            new_theta -= np.pi * 2  # make the theta still in the range
        if new_theta < -np.pi:
            new_theta += np.pi * 2
        bbox3D[6] = new_theta

        predicted_theta = self.kf.x[6]
        # if the angle of two theta is not acute angle
        if np.pi / 2.0 < abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:
            self.kf.x[6] += np.pi
            if self.kf.x[6] > np.pi:
                self.kf.x[6] -= np.pi * 2  # make the theta still in the range
            if self.kf.x[6] < -np.pi:
                self.kf.x[6] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to
        # < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[6] += np.pi * 2
            else:
                self.kf.x[6] -= np.pi * 2

        # Update the bbox3D
        self.kf.update(bbox3D)

        if self.kf.x[6] >= np.pi:
            self.kf.x[6] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[6] < -np.pi:
            self.kf.x[6] += np.pi * 2
        self.info = info
        self.prev_ref = self.kf.x.flatten()[: self.cfg.motion_dims]

    def predict(self, update_state: bool = True):
        """
        Advances the state vector and returns the predicted bounding box
        estimate.
        """
        self.kf.predict()
        if self.kf.x[6] >= np.pi:
            self.kf.x[6] -= np.pi * 2
        if self.kf.x[6] < -np.pi:
            self.kf.x[6] += np.pi * 2

        self.cfg.age += 1
        if self.cfg.time_since_update > 0:
            self.cfg.hit_streak = 0
        self.cfg.time_since_update += 1
        return torch.from_numpy(self.kf.x.flatten()).to(self.device)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return torch.from_numpy(self.kf.x.flatten()).to(self.device)

    def get_history(self):
        """
        Returns the history of estimates.
        """
        return torch.from_numpy(self.history).to(self.device)
