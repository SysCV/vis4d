"""Pointwise transformations."""
from __future__ import annotations

import math

import torch

from vis4d.data.const import CommonKeys

from .base import Transform

# Data Key that contains the bounds of a pointcloud.
PC_BOUND_KEY = "pc_bounds"


def _get_rotation_matrix(angle: float, axis: int = 0) -> torch.Tensor:
    """Creates a 3x3 Rotation Matrix.

    Args:
        angle: Rotation angle
        axis: Rotation axis

    Returns:
        3x3 Rotation Matrix

    Raises:
        ValueError: if the axis is invalid
    """
    if axis == 0:
        return torch.tensor(
            [
                [1, 0, 0],
                [0, math.cos(angle), -math.sin(angle)],
                [0, math.sin(angle), math.cos(angle)],
            ]
        )
    if axis == 1:
        return torch.tensor(
            [
                [math.cos(angle), 0, math.sin(angle)],
                [0, 1, 0],
                [-math.sin(angle), 0, math.cos(angle)],
            ]
        )
    if axis == 2:
        return torch.tensor(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
        )
    raise ValueError(f"Unknown axis {axis}")


@Transform(
    in_keys=(CommonKeys.points3d,),
    out_keys=(CommonKeys.points3d,),
)
def move_pts_to_last_channel():
    """Permutes the last two channels of the data for the given keys.

    Use this to convert a dataset sample of shape [B, n_pts, n_feat]
    to [B, n_feat, n_pts].
    """

    def _move_features_to_last_channel(*args: torch.Tensor):
        if len(args) == 1:
            return args[0].transpose(-1, -2).contiguous()
        return [d.transpose(-1, -2).contiguous() for d in args]

    return _move_features_to_last_channel


@Transform(
    in_keys=(CommonKeys.points3d, CommonKeys.colors3d),
    out_keys=(CommonKeys.points3d,),
)
def concatenate_point_features():
    """Concatenates all given data keys along the first axis."""

    def _concatenate_point_features(*args: torch.Tensor):
        return torch.cat(args)

    return _concatenate_point_features


@Transform(
    in_keys=(CommonKeys.points3d,),
    out_keys=(CommonKeys.points3d,),
)
def add_norm_noise(std: float = 0.1):
    """Adds random normal distributed noise with given std to the data.

    Args:
        std (float): Standard Deviation of the noise
    """

    def _add_norm_noise(coordinates: torch.Tensor):
        coordinates += torch.randn(coordinates.shape) * std
        return coordinates

    return _add_norm_noise


@Transform(in_keys=(CommonKeys.points3d,), out_keys=(CommonKeys.points3d,))
def add_uniform_noise(min_value: float = -0.1, max_value: float = 0.1):
    """Adds uniform distributed noise with given limits to the data.

    Args:
        min_value (float): min noise values
        max_value (float): max noise values
    """

    def _add_uniform_noise(coordinates: torch.Tensor):
        noise = torch.rand(coordinates.shape)
        noise = noise * (max_value - min_value) + min_value
        coordinates += noise
        return coordinates

    return _add_uniform_noise


@Transform(in_keys=(CommonKeys.points3d,), out_keys=(CommonKeys.points3d,))
def rotate_around_axis(angle_min=-torch.pi, angle_max=torch.pi, axis=2):
    """Rotates the given data around a specified axis.

    Args:
        angle_min (float): min rotation angle
        angle_max (float): max_rotation angle
        axis (int): around which axis to rotate.
    """

    def _rotate_around_axis(coordinates: torch.Tensor):
        angle = torch.rand(1).item()
        angle = angle * (angle_max - angle_min) + angle_min
        rotation_matrix = _get_rotation_matrix(angle, axis)
        rotated = (rotation_matrix @ coordinates.T).T
        return rotated

    return _rotate_around_axis


@Transform(in_keys=(CommonKeys.points3d,), out_keys=(CommonKeys.points3d,))
def center_and_normalize(normalize=True):
    """Centers the data and divides it by the max value.

    Args:
        normalize(bool): If true, divides the coordinates by the max values
                         in each direction.
    """

    def _center_and_normalize(coordinates: torch.Tensor):
        center = (
            coordinates.max(dim=0).values - coordinates.min(dim=0).values
        ) / 2 + coordinates.min(dim=0).values
        coords = coordinates - center
        if normalize:
            max_val = torch.max(torch.max(torch.abs(coords), dim=0).values)
            return (coords) / (max_val)

        return coords

    return _center_and_normalize


@Transform(in_keys=(CommonKeys.points3d,), out_keys=(PC_BOUND_KEY,))
def extract_pc_bounds():
    """Extracts the max and min values of the loaded points.

    The value is safed into the out_key (default 'pc_bounds').
    """

    def _extract_pc_bounds(coordinates: torch.Tensor):
        return torch.cat(
            [
                torch.min(coordinates, dim=0, keepdim=True).values,
                torch.max(coordinates, dim=0, keepdim=True).values,
            ]
        )

    return _extract_pc_bounds


@Transform(
    in_keys=(CommonKeys.points3d, PC_BOUND_KEY),
    out_keys=(CommonKeys.points3d,),
)
def normalize_by_bounds(axes: tuple[int, ...] = (0, 1)):
    """Uses the bounds stored in in_keys[1] (default 'pc_bounds').

    to normalize the data along al axes.

    Args:
        axes (tuple(int)): Axis to apply normalization for.
            Default ignores z axis.
    """

    def _extract_pc_bounds(
        coordinates: torch.Tensor, bounds: torch.Tensor
    ) -> torch.Tensor:
        max_bounds = torch.max(torch.abs(bounds), dim=0).values
        coords = coordinates
        coords[axes] = coords[axes] / max_bounds
        return coords

    return _extract_pc_bounds
