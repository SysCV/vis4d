"""Resize augmentation."""
import random
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from vis4d.struct_to_revise.structures import DictStrAny

from ..datasets.base import DataKeys, DictData
from .base import BaseBatchTransform, BaseTransform


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


class PointSampler(BaseTransform):
    """Base class for 3D point sampling operations."""

    def __init__(
        self,
        n_pts: int,
        in_keys: Tuple[str, ...] = (
            DataKeys.colors3d,
            DataKeys.points3d,
            DataKeys.semantics3d,
        ),
    ):
        """Creates a new BasePointSampler transform.
        Args:
            n_pts (int): Number of points to sample
            in_keys tuple(str): Tuple of keys on which the operation should be
            applied
        """
        super().__init__(in_keys)
        self.n_pts = n_pts

    def generate_parameters(self, data: DictData) -> DictStrAny:
        """Generate current parameters."""
        return dict()


class RandomPointSampler(PointSampler):
    """Samples points unifromly random."""

    def __call__(self, data: DictData, parameters: DictStrAny) -> DictData:
        """Apply point sampling."""
        data_out = data
        selected_idxs = None
        for in_key in self.in_keys:
            if not in_key in data:
                continue

            if selected_idxs is None:
                selected_idxs = sample_indices(self.n_pts, data[in_key])

            data_out[in_key] = data[in_key][selected_idxs, ...]

        return data_out


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


class RandomBlockPointSampler(PointSampler):
    """Samples points uniformly random inside a block around a point."""

    def __init__(
        self,
        n_pts: int,
        min_pts: 1024,
        block_size: List[float] = [1.0, 1.0, 1.0],
        coord_key: str = DataKeys.points3d,
        in_keys: Tuple[str, ...] = (
            DataKeys.colors3d,
            DataKeys.points3d,
            DataKeys.semantics3d,
        ),
        max_tries=100,
    ):
        """Selectes a block of size [block_size[0], block_size[1], +inf] at a random point and up/downsamples the points inside it.
        TODO explain more"""
        super().__init__(n_pts, in_keys)
        self.coord_key = coord_key
        self.block_size = torch.tensor(block_size)
        self.min_pts = min_pts
        self._max_tries = max_tries

    def generate_parameters(self, data: DictData) -> DictStrAny:
        """Generate current parameters. TODO"""
        coords = data[self.coordinate_key]
        coord_min, coord_max = (
            torch.min(coords, dim=0).values,
            torch.max(coords, axis=0).values,
        )
        return dict(coord_max=coord_max, coord_min=coord_min)

    def __call__(self, data: DictData, parameters: DictStrAny) -> DictData:
        """Apply point sampling."""
        data_out = data
        selected_idxs = None
        for in_key in self.in_keys:
            if not in_key in data:
                continue

            if selected_idxs is None:
                coords = data[self.coord_key]
                for _ in range(self._max_tries):
                    center_pt_idx = torch.randperm(coords.shape[0])[0]
                    center_pt = coords[center_pt_idx, ...]
                    n_pts, selected_idxs = sample_from_block(
                        self.n_pts, coords, center_pt, self.block_size
                    )
                    # Found enough points
                    if n_pts >= self.min_pts:
                        break

            data_out[in_key] = data[in_key][selected_idxs, ...]

        return data_out


class FullCoverageBlockSampler(PointSampler):
    """Samples all points in the environment using a sliding box."""

    def __init__(
        self,
        coordinate_key=DataKeys.points3d,
        min_pts_per_block=8,
        block_size=[1.0, 1.0, 1.0],
        n_pts_per_block: int = 4096,
        stride: int = 1,
        center_blocks=True,
        in_keys: Tuple[str, ...] = (
            DataKeys.colors3d,
            DataKeys.points3d,
            DataKeys.semantics3d,
        ),
    ) -> None:
        super().__init__(n_pts_per_block, sorted(in_keys))
        self.n_pts_per_block = n_pts_per_block
        self.center_blocks = center_blocks
        self.stride = stride
        self.coordinate_key = coordinate_key
        self.block_size = torch.tensor(block_size)
        self.min_pts_per_block = min_pts_per_block

    def generate_parameters(self, data: DictData) -> DictStrAny:
        """Generate current parameters."""
        coords = data[self.coordinate_key]
        coord_min, coord_max = (
            torch.min(coords, dim=0).values,
            torch.max(coords, axis=0).values,
        )
        return dict(coord_max=coord_max, coord_min=coord_min)

    def __call__(
        self, data: List[DictData], parameters: DictStrAny
    ) -> DictData:
        """Exectues the sampling."""
        data_out = data

        # Get bounding box for sampling
        coords = data[self.coordinate_key]
        coord_min, coord_max = parameters["coord_min"], parameters["coord_max"]
        hwl = coord_max - coord_min
        grid_idxs = (
            torch.ceil((hwl - self.block_size / 2.0) / self.stride) + 1
        ).int()

        sampled_coords = torch.zeros((0, coords.shape[1]), dtype=coords.dtype)

        other_sampled_pts = {}
        for key in self.in_keys:
            if key == self.coordinate_key:
                continue
            t = data[key]
            other_sampled_pts[key] = torch.zeros(
                (0, t.shape[1]), dtype=t.dtype
            )

        ns = 0
        for idx_x in range(grid_idxs[0].item()):
            for idx_y in range(grid_idxs[1].item()):
                center_pt = torch.tensor(
                    [
                        coord_min[0] + idx_x * self.stride,
                        coord_min[1] + idx_y * self.stride,
                        0,
                    ],
                    dtype=coords.dtype,
                )

                n_pts, sampled_idxs = sample_from_block(
                    self.n_pts_per_block,
                    coords,
                    center_pt,
                    block_size=self.block_size,
                )
                if (
                    n_pts < self.min_pts_per_block
                ):  # Not enough points in this block
                    continue
                sampled_coords = torch.vstack(
                    [sampled_coords, coords[sampled_idxs, ...]]
                )

                for key in self.in_keys:
                    if key == self.coordinate_key:
                        continue
                    t = data[key]
                    other_sampled_pts[key] = torch.vstack(
                        [other_sampled_pts[key], t[sampled_idxs, ...]]
                    )

                ns += n_pts

        for key in self.in_keys:
            if key == self.coordinate_key:
                data_out[key] = sampled_coords
            else:
                data_out[key] = other_sampled_pts[key]

        return data
