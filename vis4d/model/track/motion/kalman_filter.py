"""Kalman Filter implementation based on torch for vis4d."""
from typing import Optional, Tuple

import torch
from torch import nn


class KalmanFilter(nn.Module):  # type: ignore # pylint: disable=abstract-method
    """Kalman filter.

    A general Kalman filter for tracking bounding boxes in image space.
    Suppose measurement is a N-dimensional state space.

    Args:
        motion_mat: motion matrix with shape 2Nx2N.
        update_mat: update matrix with shape Nx2N.
        cov_motion_Q: covariance matrix with shape 2Nx2N.
        cov_project_R: covariance matrix with shape NxN.
        cov_P0: covariance matrix with shape 2Nx2N.
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
        self.register_buffer("_motion_mat", motion_mat, False)
        self.register_buffer("_update_mat", update_mat, False)
        self.register_buffer("_cov_motion_Q", cov_motion_Q, False)
        self.register_buffer("_cov_project_R", cov_project_R, False)
        self.register_buffer("_cov_P0", cov_P0, False)

    def initiate(
        self,
        measurement: torch.Tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Initiate a Kalman filter state based on the first measurement.

        Args:
            measurement: N-dimensional state space.
            In normal case, it represents bounding box coordinates (
            x, y, a, h) with center
                position (x, y), aspect ratio a, and height h.

        Returns:
            mean, covariance: the mean vector (2N dimensional) and covariance
                matrix (2Nx2N dimensional) of the new track. Unobserved
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
            mean: The 2N dimensional mean vector of the object state at the
                previous time step.
            covariance: The 2Nx2N dimensional covariance matrix of the object
                state at the previous time step.

        Returns:
            mean: the mean vector, Unobserved velocities are initialized to
                0 mean.
            covariance: covariance matrix of the predicted state.
        """
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
                The state's mean vector (2N dimensional vector).
            covariance :
                The state's covariance matrix (2Nx2N dimensional).

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
                The predicted state's mean vector (2N dimensional).
            covariance :
                The state's covariance matrix (2Nx2N dimensional).
            measurement :
                The 4 dimensional measurement vector (N dimensional),
                N is measurement state size.

        Returns:
            new_mean: updated mean (2N dimensional)
            new_covariance: updated covariance (2Nx2N dimensional).
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
                Mean vector over the state distribution (2N dimensional).
            covariance :
                Covariance of the state distribution (2Nx2N dimensional).
            measurements :
                A N dimensional measurements, And in normal case. It represents
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
        if only_position:  # pragma: no cover
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
