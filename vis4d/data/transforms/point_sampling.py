"""Contains Different Sampling methods to downsample pointclouds."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from vis4d.data.const import CommonKeys

from .base import Transform


def sample_indices(n_pts: int, data: torch.Tensor) -> torch.Tensor:
    """Samples n_pts indices from the first dim of the provided data tensor.

    Args:
        n_pts (int): Number of indices to sample
        data (Tensor): Data from which to sample indices

    Raises:
        ValueError: If data is empty.

    Returns:
        torch Tensor containing the sampled indices.
    """
    if len(data) == 0:
        raise ValueError("Data sample was empty!")

    perm = torch.randperm(data.size(0))
    selected_idxs = perm[:n_pts]
    if len(selected_idxs) != n_pts:
        n_to_sample = n_pts - len(selected_idxs)
        return torch.concat(
            [
                selected_idxs,
                torch.randint(data.size(0), (n_to_sample, 1)).reshape(-1),
            ]
        )

    return selected_idxs


def sample_from_block(
    n_pts: int,
    data: torch.Tensor,
    center_xyz: torch.Tensor,
    block_size: torch.Tensor,
    ignore_axis: Sequence[int] = (2,),
) -> tuple[int, torch.Tensor]:
    """Samples point indices inside a box.

    Args:
        n_pts (int): How many points to sample
        data: (Tensor): Data containing pointwise information. Shape [n_pts, x]
        center_xyz (Tensor): Center point around which to sample (x,y,z) [3]
        block_size (Tensor): Block length in each direction (x,y,z) [3]
        ignore_axis (Sequence[int]): If specified, this axis will be ignored
            and all points along this axis will be considered

    Returns:
        tuple[int, Tensor]: Number of points that were in the box and
                            the selected indices of shape [n_pts]
    """
    min_data = torch.min(data, dim=0).values
    max_data = torch.max(data, dim=0).values
    max_box = center_xyz + block_size / 2.0
    min_box = center_xyz - block_size / 2.0
    for axis in ignore_axis:
        max_box[axis] = max_data[axis]
        min_box[axis] = min_data[axis]

    box_mask = torch.logical_and(
        torch.all(data >= min_box, dim=1), torch.all(data <= max_box, dim=1)
    )
    if box_mask.sum().item() == 0:  # No valid data sample found!
        return 0, torch.tensor([], dtype=torch.int)

    selected_idxs_masked = sample_indices(n_pts, data[box_mask, ...])

    masked_idxs = torch.arange(data.shape[0])[box_mask]
    selected_idxs_global = masked_idxs[selected_idxs_masked]
    return int(torch.sum(box_mask).item()), selected_idxs_global


@Transform(
    in_keys=(CommonKeys.points3d,),
    out_keys=(CommonKeys.points3d,),
)
def sample_points_random(num_pts: int = 1024):
    """Subsamples points randomly.

    Samples 'num_pts' (with repetition if needed) randomly from the
    provided data tensors.

    Args:
        num_pts (int): How many points to sample
    """

    def _sample_points_random(
        coordinates: torch.Tensor, *args: torch.Tensor
    ) -> torch.Tensor | list[torch.Tensor]:
        selected_idxs = sample_indices(num_pts, coordinates)
        sampled_coords = coordinates[selected_idxs, ...]
        if len(args) == 0:
            return sampled_coords
        return [sampled_coords] + [d[selected_idxs, ...] for d in args]

    return _sample_points_random


@Transform(in_keys=(CommonKeys.points3d,), out_keys=(CommonKeys.points3d,))
def sample_points_block_random(
    num_pts: int = 1024,
    min_pts: int = 32,
    block_size: Sequence[float] = (1.0, 1.0, 1.0),
    max_tries: int = 100,
    center: bool = False,
):
    """Subsamples points around a randomly chosen block.

    Samples 'num_pts' (with repetition if needed) around a random block from
    the provided data tensors.

    Args:
        num_pts (int): How many points to sample
        min_pts (int): Only sample points if at least these many points
                       are inside it
        block_size (Sequence[float]): Dimension of block from which to sample
            (xyz)
        max_tries (bool): Maximum of tries to sample a block before giving up
        center (bool): If true, substracts center point coordiantes
    """

    def _sample_points_block_random(
        coordinates: torch.Tensor, *args: torch.Tensor
    ) -> torch.Tensor | list[torch.Tensor]:
        """Apply point sampling."""
        for _ in range(max_tries):
            center_pt_idx = torch.randperm(coordinates.shape[0])[0]
            center_pt = coordinates[center_pt_idx, ...]
            n_pts, selected_idxs = sample_from_block(
                num_pts, coordinates, center_pt, torch.tensor(block_size)
            )
            # Found enough points
            if n_pts >= min_pts:
                break
        sampled_coords = coordinates[selected_idxs, ...]
        if center:
            sampled_coords -= center_pt

        if len(args) == 0:
            return sampled_coords
        return [sampled_coords] + [d[selected_idxs, ...] for d in args]

    return _sample_points_block_random


@Transform(
    in_keys=(CommonKeys.points3d,),
    out_keys=(CommonKeys.points3d,),
)
def sample_points_block_full_coverage(  # pylint: disable=invalid-name
    n_pts_per_block: int = 1024,
    min_pts_per_block: int = 1,
    block_size: Sequence[float] = (1.0, 1.0, 1.0),
):
    """Subsamples the full pointcloud by regularly dividing it into blocks.

    Divides a room into boxes of size 'block size' and samples 'num_pts'
    (with repetition if needed from the provided data tensors for each block.
    Boxes are only sampled at the xy plane.

    Args:
        n_pts_per_block (int): How many points to sample per block.
        min_pts_per_block (int): Only sample points if at least these many
            points are inside the block.
        block_size (Sequence[float]): Dimension of block from which to
            sample (xyz).
    """

    def _sample_points_block_full_coverage(  # pylint: disable=invalid-name
        coordinates: torch.Tensor, *args: torch.Tensor
    ) -> torch.Tensor | list[torch.Tensor]:
        """Apply point sampling."""
        # Get bounding box for sampling
        coord_min, coord_max = (
            torch.min(coordinates, dim=0).values,
            torch.max(coordinates, dim=0).values,
        )
        hwl = coord_max - coord_min

        block_size_torch = torch.tensor(block_size)

        grid_idxs = (
            torch.ceil((hwl - torch.tensor(block_size) / 2.0)) + 1
        ).int()

        sampled_coords = torch.zeros(
            (0, coordinates.shape[1]), dtype=coordinates.dtype
        )

        other_sampled_pts = []
        for d in args:
            other_sampled_pts.append(
                torch.zeros((0, d.shape[1]), dtype=d.dtype)
                if len(d.shape) > 1
                else torch.zeros((0), dtype=d.dtype)
            )

        for idx_x in range(grid_idxs[0].item()):
            for idx_y in range(grid_idxs[1].item()):
                center_pt = torch.tensor(
                    [
                        coord_min[0] + idx_x,
                        coord_min[1] + idx_y,
                        0,
                    ],
                    dtype=coordinates.dtype,
                )

                n_pts, sampled_idxs = sample_from_block(
                    n_pts_per_block,
                    coordinates,
                    center_pt,
                    block_size=block_size_torch,
                )
                if (
                    n_pts < min_pts_per_block
                ):  # Not enough points in this block
                    continue
                sampled_coords = torch.vstack(
                    [sampled_coords, coordinates[sampled_idxs, ...]]
                )

                for idx, data in enumerate(other_sampled_pts):
                    other_sampled_pts[idx] = torch.cat(
                        [data, args[idx][sampled_idxs, ...]], dim=0
                    )

        if len(args) == 0:
            return sampled_coords
        return [sampled_coords] + other_sampled_pts

    return _sample_points_block_full_coverage
