from typing import Dict, List, Optional, Tuple
import torch

import scipy.linalg


class KalmanFilter(object):
    """Kalman Filter class.

    The 8-dimensional state space,
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    """

    def __init__(self):
        """Init."""
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = torch.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = torch.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = torch.zeros_like(mean_pos)
        mean = torch.cat([mean_pos, mean_vel])

        std = torch.Tensor(
            [
                2 * self._std_weight_position * measurement[3],
                2 * self._std_weight_position * measurement[3],
                1e-2,
                2 * self._std_weight_position * measurement[3],
                10 * self._std_weight_velocity * measurement[3],
                10 * self._std_weight_velocity * measurement[3],
                1e-5,
                10 * self._std_weight_velocity * measurement[3],
            ]
        )
        covariance = torch.diag(torch.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step."""
        std_pos = torch.Tensor(
            [
                self._std_weight_position * mean[3],
                self._std_weight_position * mean[3],
                1e-2,
                self._std_weight_position * mean[3],
            ]
        )
        std_vel = torch.Tensor(
            [
                self._std_weight_velocity * mean[3],
                self._std_weight_velocity * mean[3],
                1e-5,
                self._std_weight_velocity * mean[3],
            ]
        )
        motion_cov = torch.diag(torch.square(torch.cat((std_pos, std_vel))))

        mean = torch.matmul(self._motion_mat, mean)
        covariance = (
            torch.matmul(
                self._motion_mat, torch.matmul(covariance, self._motion_mat.T)
            )
            + motion_cov
        )

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space."""
        std = torch.Tensor(
            [
                self._std_weight_position * mean[3],
                self._std_weight_position * mean[3],
                1e-1,
                self._std_weight_position * mean[3],
            ]
        )
        innovation_cov = torch.diag(torch.square(std))

        mean = torch.matmul(self._update_mat, mean)
        covariance = torch.matmul(
            self._update_mat, torch.matmul(covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step."""
        projected_mean, projected_cov = self.project(mean, covariance)
        projected_cov = projected_cov.numpy()
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            torch.matmul(covariance, self._update_mat.T).numpy().T,
            check_finite=False,
        ).T
        kalman_gain = torch.from_numpy(kalman_gain)
        innovation = measurement - projected_mean

        new_mean = mean + torch.matmul(innovation, kalman_gain.T)
        projected_cov = torch.from_numpy(projected_cov)
        new_covariance = covariance - torch.matmul(
            kalman_gain, torch.matmul(projected_cov, kalman_gain.T)
        )

        return new_mean, new_covariance
