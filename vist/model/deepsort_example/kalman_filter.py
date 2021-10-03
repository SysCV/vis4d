"""Kalman filter."""
from typing import Optional, Tuple

import torch
from torch import nn


# Table for the 0.95 quantile of the chi-square distribution with N degrees of
# freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
# function and used as Mahalanobis gating threshold.

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilter(nn.Module):  # type: ignore
    """Kalman filter.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    """

    def __init__(
        self,
        motion_mat: torch.Tensor,
        update_mat: torch.Tensor,
        cov_motion_Q: torch.Tensor,
        cov_project_R: torch.Tensor,
        cov_P0: torch.Tensor,
    ) -> None:
        """Init."""
        super().__init__()
        self.register_buffer("_motion_mat", motion_mat)
        self.register_buffer("_update_mat", update_mat)
        self.register_buffer("_cov_motion_Q", cov_motion_Q)
        self.register_buffer("_cov_project_R", cov_project_R)
        self.register_buffer("_cov_P0", cov_P0)

    def initiate(
        self,
        measurement: torch.Tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Initiate a Kalman filter state based on the first measurement

        Args:
            measurement: Bounding box coordinates (x, y, a, h) with center
                position (x, y), aspect ratio a, and height h.

        Returns:
            mean, covariance: the mean vector (8 dimensional) and covariance
                matrix (8x8 dimensional) of the new track. Unobserved
                velocities are initialized to 0 mean.
        """
        mean_pos = measurement.clone().detach()
        mean_vel = torch.zeros_like(mean_pos)
        mean = torch.cat([mean_pos, mean_vel], dim=0)

        covariance = self._cov_P0
        return mean, covariance

    def predict(
        self,
        mean: torch.tensor,
        covariance: torch.tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Run Kalman filter prediction step.

        Args:
            mean: The 8 dimensional mean vector of the object state at the
                previous time step.
            covariance: The 8x8 dimensional covariance matrix of the object
                state at the previous time step.

        Returns:
            mean: the mean vector, Unobserved velocities are initialized to
                0 mean.
            covariance: covariance matrix of the predicted state.
        """
        print("_motion_mat: ", self._motion_mat.device)
        print("mean: ", mean)
        mean = torch.matmul(self._motion_mat, mean)
        covariance = (
            torch.matmul(
                self._motion_mat, torch.matmul(covariance, self._motion_mat.T)
            )
            + self._cov_motion_Q
        )

        return mean, covariance

    def project(
        self, mean: torch.tensor, covariance: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Project state distribution to measurement space.

        Args:
            mean :
                The state's mean vector (8 dimensional vector).
            covariance :
                The state's covariance matrix (8x8 dimensional).

        Returns:
            mean: the projected mean of the given state estimate.
            projected_cov: the projected covariance matrix of the given state
                estimate.
        """
        mean = torch.matmul(self._update_mat, mean)
        covariance = torch.matmul(
            self._update_mat, torch.matmul(covariance, self._update_mat.T)
        )
        projected_cov = covariance + self._cov_project_R
        return mean, projected_cov

    def update(
        self,
        mean: torch.tensor,
        covariance: torch.tensor,
        measurement: torch.tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Run Kalman filter correction step.

        Args:
            mean :
                The predicted state's mean vector (8 dimensional).
            covariance :
                The state's covariance matrix (8x8 dimensional).
            measurement :
                The 4 dimensional measurement vector (x, y, a, h), where (x, y)
                is the center position, a the aspect ratio, and h the height of
                the bounding box.

        Returns:
            new_mean: updated mean
            new_covariance: updated covariance
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor = torch.cholesky(projected_cov)
        kalman_gain = torch.cholesky_solve(
            torch.matmul(covariance, self._update_mat.T).T,
            chol_factor,
            upper=False,
        ).T

        innovation = measurement - projected_mean

        new_mean = mean + torch.matmul(innovation, kalman_gain.T)
        new_covariance = covariance - torch.matmul(
            kalman_gain, torch.matmul(projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: torch.tensor,
        covariance: torch.tensor,
        measurements: torch.tensor,
        only_position: Optional[bool] = False,
    ) -> torch.tensor:
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Args:
            mean :
                Mean vector over the state distribution (8 dimensional).
            covariance :
                Covariance of the state distribution (8x8 dimensional).
            measurements :
                An Nx4 dimensional matrix of N measurements, each in
                format (x, y, a, h) where (x, y) is the bounding box center
                position, a the aspect ratio, and h the height.
            only_position: If True, distance computation is done with respect
                to the bounding box center position only.

        Returns:
            squared_maha: a vector of length N, where the i-th element contains
                the squared Mahalanobis distance between (mean, covariance) and
                `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        cholesky_factor = torch.cholesky(covariance)
        d = measurements - mean
        z = torch.triangular_solve(
            d.T,
            cholesky_factor,
            upper=False,
        )[0]
        squared_maha = torch.sum(z * z, axis=0)
        return squared_maha
