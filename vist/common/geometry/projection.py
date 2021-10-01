"""Projection utilities."""
import torch

from vist.struct import Intrinsics


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
        intrinsic_matrix = intrinsics.tensor.squeeze(0).T
    elif len(hom_coords.shape) == 3:
        intrinsic_matrix = intrinsics.tensor.permute(0, 2, 1)
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
        inv_intrinsics = torch.inverse(intrinsics.tensor).squeeze(0).T
        if len(depths.shape) == 1:
            depths = depths.unsqueeze(-1)
        assert len(depths.shape) == 2, "depths must have same dims as points"
    elif len(points.shape) == 3:
        inv_intrinsics = torch.inverse(intrinsics.tensor).permute(0, 2, 1)
        if len(depths.shape) == 2:
            depths = depths.unsqueeze(-1)
        assert len(depths.shape) == 3, "depths must have same dims as points"
    else:
        raise ValueError(f"Shape of input points not valid: {points.shape}")
    hom_coords = torch.cat([points, torch.ones_like(points)[..., 0:1]], -1)
    pts_3d = hom_coords @ inv_intrinsics
    pts_3d *= depths
    return pts_3d
