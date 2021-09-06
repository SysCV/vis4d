"""Kalman filter."""
from typing import Optional, Tuple

import torch

from .kf_parameters import cov_motion_Q, cov_P0, cov_project_R

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


class KalmanFilter:
    """Kalman filter.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    """

    def __init__(self):
        """Init."""
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = torch.eye(2 * ndim, 2 * ndim).cuda()
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = torch.eye(ndim, 2 * ndim).cuda()
        self.idx2cls_mapping = {
            0: "pedestrian",
            1: "rider",
            2: "car",
            3: "truck",
            4: "bus",
            5: "train",
            6: "motorcycle",
            7: "bicycle",
        }

    def initiate(
        self, measurement: torch.tensor, class_id: int
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Create track from unassociated measurement.

        Args:
            measurement: Bounding box coordinates (x, y, a, h) with center
                position (x, y), aspect ratio a, and height h.
            class_id: class id of this detection measurement

        Returns:
            mean, covariance: the mean vector (8 dimensional) and covariance
                matrix (8x8 dimensional) of the new track. Unobserved
                velocities are initialized to 0 mean.
        """
        mean_pos = measurement.clone().detach()
        mean_vel = torch.zeros_like(mean_pos)
        mean = torch.cat([mean_pos, mean_vel], dim=0)

        covariance = cov_P0[self.idx2cls_mapping[class_id]].to(mean_pos.device)
        return mean, covariance

    def predict(
        self,
        mean: torch.tensor,
        covariance: torch.tensor,
        class_id: torch.tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Run Kalman filter prediction step.

        Args:
            mean: The 8 dimensional mean vector of the object state at the
                previous time step.
            covariance: The 8x8 dimensional covariance matrix of the object
                state at the previous time step.
            class_id: class id of this detection measurement

        Returns:
            mean: the mean vector, Unobserved velocities are initialized to
                0 mean.
            covariance: covariance matrix of the predicted state.
        """
        motion_cov = cov_motion_Q[self.idx2cls_mapping[class_id]].to(
            mean.device
        )
        mean = torch.matmul(self._motion_mat, mean)
        covariance = (
            torch.matmul(
                self._motion_mat, torch.matmul(covariance, self._motion_mat.T)
            )
            + motion_cov
        )

        return mean, covariance

    def project(
        self, mean: torch.tensor, covariance: torch.tensor, class_id: int
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Project state distribution to measurement space.

        Args:
            mean :
                The state's mean vector (8 dimensional vector).
            covariance :
                The state's covariance matrix (8x8 dimensional).
            class_id :
                class id of this detection measurement

        Returns:
            mean: the projected mean of the given state estimate.
            projected_cov: the projected covariance matrix of the given state
                estimate.
        """
        innovation_cov = cov_project_R[self.idx2cls_mapping[class_id]].to(
            mean.device
        )

        mean = torch.matmul(self._update_mat, mean)
        covariance = torch.matmul(
            self._update_mat, torch.matmul(covariance, self._update_mat.T)
        )
        projected_cov = covariance + innovation_cov
        return mean, projected_cov

    def update(
        self,
        mean: torch.tensor,
        covariance: torch.tensor,
        measurement: torch.tensor,
        class_id: int,
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
            class_id : int
                class id of this detection measurement

        Returns:
            new_mean: updated mean
            new_covariance: updated covariance
        """
        projected_mean, projected_cov = self.project(
            mean, covariance, class_id
        )

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
        class_id: int,
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
            class_id : class id of this detection measurement

        Returns:
            squared_maha: a vector of length N, where the i-th element contains
                the squared Mahalanobis distance between (mean, covariance) and
                `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance, class_id)
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
