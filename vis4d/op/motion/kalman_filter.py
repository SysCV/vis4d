"""Kalman Filter PyTorch implementation."""

from __future__ import annotations

import torch
from torch import Tensor


def predict(
    motion_mat: Tensor,
    cov_motion_q: Tensor,
    mean: Tensor,
    covariance: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run Kalman filter prediction step."""
    # x = Fx
    mean = torch.matmul(motion_mat, mean)

    # P = (FP)F + Q
    covariance = (
        torch.matmul(motion_mat, torch.matmul(covariance, motion_mat.T))
        + cov_motion_q
    )

    return mean, covariance


def project(
    update_mat: Tensor, cov_project_r: Tensor, mean: Tensor, covariance: Tensor
) -> tuple[Tensor, Tensor]:
    """Project state distribution to measurement space."""
    # Hx
    mean = torch.matmul(update_mat, mean)

    # HPH^T + R
    covariance = torch.matmul(
        update_mat, torch.matmul(covariance, update_mat.T)
    )
    projected_cov = covariance + cov_project_r
    return mean, projected_cov


def update(
    update_mat: Tensor,
    cov_project_r: Tensor,
    mean: Tensor,
    covariance: Tensor,
    measurement: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run Kalman filter correction step."""
    # Hx, S = HPH^T + R
    projected_mean, projected_cov = project(
        update_mat, cov_project_r, mean, covariance
    )

    # K = PHT * S^-1
    chol_factor = torch.linalg.cholesky(  # pylint: disable=not-callable
        projected_cov
    )
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
    i_kh = torch.eye(mean.shape[-1]).to(
        device=measurement.device
    ) - torch.matmul(kalman_gain, update_mat)

    new_covariance = torch.matmul(
        torch.matmul(i_kh, covariance), i_kh.T
    ) + torch.matmul(torch.matmul(kalman_gain, cov_project_r), kalman_gain.T)

    return new_mean, new_covariance
