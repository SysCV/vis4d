"""Kalman Filter motion model."""
from __future__ import annotations

from torch import Tensor

from vis4d.op.geometry.rotation import acute_angle, normalize_angle

from .kalman_filter import predict, update


def kf3d_update(
    update_mat: Tensor,
    cov_project_R: Tensor,
    mean: Tensor,
    covariance: Tensor,
    obs_3d: Tensor,
) -> tuple[Tensor, Tensor]:
    """KF3D update function."""
    mean[6] = normalize_angle(mean[6])
    obs_3d[6] = normalize_angle(obs_3d[6])

    mean[6] = acute_angle(mean[6], obs_3d[6])

    new_mean, new_covariance = update(
        update_mat,
        cov_project_R,
        mean,
        covariance,
        obs_3d,
    )
    new_mean[6] = normalize_angle(new_mean[6])
    return new_mean, new_covariance


def kf3d_predict(
    motion_mat: Tensor, cov_motion_Q: Tensor, mean: Tensor, covariance: Tensor
) -> tuple[Tensor, Tensor]:  # TODO: check update_state=motion_model.age != 0
    """KF3D predict function."""
    new_mean, new_covariance = predict(
        motion_mat, cov_motion_Q, mean, covariance
    )
    new_mean[6] = normalize_angle(new_mean[6])
    return new_mean, new_covariance
