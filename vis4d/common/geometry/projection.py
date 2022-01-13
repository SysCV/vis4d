"""Projection utilities."""
from typing import Optional, Tuple

import torch

from vis4d.struct import Intrinsics


def project_points(
    points: torch.Tensor, intrinsics: Intrinsics
) -> torch.Tensor:
    """Project points to pixel coordinates with given intrinsics.

    Args:
        points: (N, 3) or (B, N, 3) 3D coordinates.
        intrinsics: Intrinsics class with 1 entry or B entries.

    Returns:
        torch.Tensor: (N, 2) or (B, N, 2) 2D pixel coordinates.

    Raises:
        ValueError: Shape of input points is not valid for computation.
    """
    assert points.shape[-1] == 3, "Input coordinates must be 3 dimensional!"
    hom_coords = points / points[..., 2:3]
    if len(hom_coords.shape) == 2:
        assert (
            len(intrinsics) == 1
        ), "Got multiple intrinsics for single point set!"
        intrinsic_matrix = intrinsics.transpose().tensor.squeeze(0)
    elif len(hom_coords.shape) == 3:
        intrinsic_matrix = intrinsics.transpose().tensor
    else:
        raise ValueError(f"Shape of input points not valid: {points.shape}")
    pts_2d = hom_coords @ intrinsic_matrix
    return pts_2d[..., :2]


def unproject_points(
    points: torch.Tensor, depths: torch.Tensor, intrinsics: Intrinsics
) -> torch.Tensor:
    """Un-projects pixel coordinates to 3D coordinates with given intrinsics.

    Args:
        points: (N, 2) or (B, N, 2) 2D pixel coordinates.
        depths: (N,) / (N, 1) or (B, N,) / (B, N, 1) depth values.
        intrinsics: Intrinsics class with 1 entry or B entries.

    Returns:
        torch.Tensor: (N, 3) or (B, N, 3) 3D coordinates.

    Raises:
        ValueError: Shape of input points is not valid for computation.
    """
    if len(points.shape) == 2:
        assert (
            len(intrinsics) == 1
        ), "Got multiple intrinsics for single point set!"
        inv_intrinsics = intrinsics.inverse().transpose().tensor.squeeze(0)
        if len(depths.shape) == 1:
            depths = depths.unsqueeze(-1)
        assert len(depths.shape) == 2, "depths must have same dims as points"
    elif len(points.shape) == 3:
        inv_intrinsics = intrinsics.inverse().transpose().tensor
        if len(depths.shape) == 2:
            depths = depths.unsqueeze(-1)
        assert len(depths.shape) == 3, "depths must have same dims as points"
    else:
        raise ValueError(f"Shape of input points not valid: {points.shape}")
    hom_coords = torch.cat([points, torch.ones_like(points)[..., 0:1]], -1)
    pts_3d = hom_coords @ inv_intrinsics
    pts_3d *= depths
    return pts_3d


def generate_projected_point_mask(
    depths: torch.Tensor,
    pts_2d: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Generate mask to filter out out range points."""
    mask = torch.ones_like(depths)
    mask = torch.logical_and(mask, depths > 0)
    mask = torch.logical_and(mask, pts_2d[:, 0] > 0)
    mask = torch.logical_and(mask, pts_2d[:, 0] < image_width - 1)
    mask = torch.logical_and(mask, pts_2d[:, 1] > 0)
    mask = torch.logical_and(mask, pts_2d[:, 1] < image_height - 1)
    return mask


def generate_depth_map(
    points_cam: torch.Tensor,
    camera_intrinsics: Intrinsics,
    image_width: int,
    image_height: int,
    pre_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate depth map."""
    pts_2d = project_points(points_cam, camera_intrinsics)
    depths = points_cam[:, 2]

    depth_map = torch.zeros((image_height, image_width)).to(points_cam.device)
    mask = generate_projected_point_mask(
        depths, pts_2d, image_width, image_height
    )
    if pre_mask is not None:
        mask &= pre_mask

    pts_2d = pts_2d[mask]
    depths = depths[mask]
    depth_map[
        pts_2d[:, 1].type(torch.long), pts_2d[:, 0].type(torch.long)
    ] = depths

    return depth_map, pts_2d, depths, mask
