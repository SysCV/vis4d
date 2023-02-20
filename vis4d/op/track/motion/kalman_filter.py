"""Kalman Filter implementation based on torch for vis4d."""
from __future__ import annotations

import torch
from torch import Tensor


def predict(
    motion_mat: Tensor,
    cov_motion_Q: Tensor,
    mean: Tensor,
    covariance: Tensor,
) -> tuple[Tensor, Tensor]:
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
    # x = Fx
    mean = torch.matmul(motion_mat, mean)

    # P = (FP)F + Q
    covariance = (
        torch.matmul(motion_mat, torch.matmul(covariance, motion_mat.T))
        + cov_motion_Q
    )

    return mean, covariance


def project(
    update_mat: Tensor, cov_project_R: Tensor, mean: Tensor, covariance: Tensor
) -> tuple[Tensor, Tensor]:
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
    # Hx
    mean = torch.matmul(update_mat, mean)

    # HPH^T + R
    covariance = torch.matmul(
        update_mat, torch.matmul(covariance, update_mat.T)
    )
    projected_cov = covariance + cov_project_R
    return mean, projected_cov


def update(
    update_mat: Tensor,
    cov_project_R: Tensor,
    mean: Tensor,
    covariance: Tensor,
    measurement: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run Kalman filter correction step.

    Args:
        mean :
            The predicted state's mean vector (2N dimensional).
        covariance :
            The state's covariance matrix (2Nx2N dimensional).
        measurement :
            The  measurement vector (N dimensional),
            N is measurement state size.

    Returns:
        new_mean: updated mean (2N dimensional)
        new_covariance: updated covariance (2Nx2N dimensional).
    """
    # Hx, S = HPH^T + R
    projected_mean, projected_cov = project(
        update_mat, cov_project_R, mean, covariance
    )

    # K = PHT * S^-1
    chol_factor = torch.linalg.cholesky(projected_cov)
    kalman_gain = torch.cholesky_solve(
        torch.matmul(covariance, update_mat.T).T,
        chol_factor,
        upper=False,
    ).T

    # y = z - Hx
    innovation = measurement - projected_mean

    # x = x + Ky
    new_mean = mean + torch.matmul(innovation, kalman_gain.T)

    # P = (I-KH)P(I-KH)' + KRK'
    # This is more numerically stable
    # and works for non-optimal K vs the equation
    # P = (I-KH)P usually seen in the literature.
    I = torch.eye(mean.shape[-1]).to(device=measurement.device)

    I_KH = I - torch.matmul(kalman_gain, update_mat)

    new_covariance = torch.matmul(
        torch.matmul(I_KH, covariance), I_KH.T
    ) + torch.matmul(torch.matmul(kalman_gain, cov_project_R), kalman_gain.T)

    return new_mean, new_covariance
