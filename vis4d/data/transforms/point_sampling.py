"""Resize augmentation."""
import random
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from vis4d.common import COMMON_KEYS

from .base import Transform


def sample_indices(n_pts: int, data: torch.tensor):
    """Samples n_pts indices from the first dimension of the provided data
    tensor.
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
    data: torch.tensor,
    center_xyz: torch.tensor,
    block_size: torch.tensor,
    ignore_axis=[2],
) -> Tuple[int, torch.tensor]:
    """
    Returns the selected indices
    """
    min_data = torch.min(data, dim=0).values
    max_data = torch.max(data, dim=0).values
    max_box = center_xyz + block_size / 2.0
    min_box = center_xyz - block_size / 2.0
    for axis in ignore_axis:
        max_box[axis] = max_data[axis]
        min_box[axis] = min_data[axis]

    box_mask = torch.logical_and(
        torch.all(data >= min_box, axis=1), torch.all(data <= max_box, axis=1)
    )
    if box_mask.sum().item() == 0:  # No valid data sample found!
        return 0, torch.tensor([], dtype=int)

    selected_idxs_masked = sample_indices(n_pts, data[box_mask, ...])

    masked_idxs = torch.arange(data.shape[0])[box_mask]
    selected_idxs_global = masked_idxs[selected_idxs_masked]
    return torch.sum(box_mask).item(), selected_idxs_global


@Transform(
    in_keys=[COMMON_KEYS.points3d],
    out_keys=[COMMON_KEYS.points3d],
)
def sample_points_random(num_pts: int = 1024):
    def _sample_points_random(
        coordinates: torch.Tensor, *args: List[torch.Tensor]
    ):
        selected_idxs = sample_indices(num_pts, coordinates)
        sampled_coords = coordinates[selected_idxs, ...]
        if len(args) == 0:
            return sampled_coords
        return [sampled_coords] + [d[selected_idxs, ...] for d in args]

    return _sample_points_random


@Transform(
    in_keys=[COMMON_KEYS.points3d],
    out_keys=[COMMON_KEYS.points3d],
)
def sample_points_block_random(
    num_pts: int = 1024,
    min_pts: int = 32,
    block_size: List[float] = [1.0, 1.0, 1.0],
    max_tries=100,
    center=False,
):
    """Assumes first key is the coordiante key!"""

    def _sample_points_block_random(coordinates, *args):
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
            sampled_coords -= torch.mean(sampled_coords, dim=0)

        if len(args) == 0:
            return sampled_coords
        return [sampled_coords] + [d[selected_idxs, ...] for d in args]

    return _sample_points_block_random


@Transform(
    in_keys=[COMMON_KEYS.points3d],
    out_keys=[COMMON_KEYS.points3d],
)
def sample_points_block_full_coverage(
    n_pts_per_block: int = 1024,
    min_pts_per_block: int = 1,
    block_size: List[float] = [1.0, 1.0, 1.0],
    stride: int = 1,
):
    """Assumes first key is the coordiante key!"""

    def _sample_points_block_full_coverage(coordinates, *args):
        """Apply point sampling."""
        # Get bounding box for sampling
        coord_min, coord_max = (
            torch.min(coordinates, dim=0).values,
            torch.max(coordinates, axis=0).values,
        )
        hwl = coord_max - coord_min

        block_size_torch = torch.tensor(block_size)

        grid_idxs = (
            torch.ceil((hwl - torch.tensor(block_size) / 2.0) / stride) + 1
        ).int()

        sampled_coords = torch.zeros(
            (0, coordinates.shape[1]), dtype=coordinates.dtype
        )

        other_sampled_pts = []
        for d in args:
            other_sampled_pts.append(
                torch.zeros((0, d.shape[1]), dtype=d.dtype)
            )

        for idx_x in range(grid_idxs[0].item()):
            for idx_y in range(grid_idxs[1].item()):
                center_pt = torch.tensor(
                    [
                        coord_min[0] + idx_x * stride,
                        coord_min[1] + idx_y * stride,
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
                    other_sampled_pts[idx] = torch.vstack(
                        [data, args[idx][sampled_idxs, ...]]
                    )

        if len(args) == 0:
            return sampled_coords
        return [sampled_coords] + other_sampled_pts

    return _sample_points_block_full_coverage
