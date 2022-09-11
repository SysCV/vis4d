"""Vis4D geometric transformation functions."""
import torch


def transform_points(
    points: torch.Tensor, transform: torch.Tensor
) -> torch.Tensor:
    """Applies transform to points.

    Args:
        points: points of shape (N, D) or (B, N, D).
        transform: transformations of shape (D+1, D+1) or (B, D+1, D+1).

    Returns:
        torch.Tensor: (N, D) / (B, N, D) transformed points.

    Raises:
        ValueError: If either points or transform have incorrect shape
    """
    hom_coords = torch.cat([points, torch.ones_like(points[..., 0:1])], -1)
    if len(points.shape) == 2:
        if len(transform.shape) == 3:
            assert (
                transform.shape[0] == 1
            ), "Got multiple transforms for single point set!"
            transform = transform.squeeze(0)
        transform = transform.T
    elif len(points.shape) == 3:
        if len(transform.shape) == 2:
            transform = transform.T.unsqueeze(0)
        elif len(transform.shape) == 3:
            transform = transform.permute(0, 2, 1)
        else:
            raise ValueError(f"Shape of transform invalid: {transform.shape}")
    else:
        raise ValueError(f"Shape of input points invalid: {points.shape}")
    points_transformed = hom_coords @ transform
    return points_transformed[..., : points.shape[-1]]
