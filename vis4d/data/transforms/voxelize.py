"""Voxelization transforms for pointclouds.

We follow openpoints implementation for voxelization.
https://github.com/guochengqian/openpoints/blob/ee100c81b1d9603c0fc76a3ee4e37d10b2af60ba/dataset/data_util.py
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np

from vis4d.common.typing import (
    NDArrayFloat,
    NDArrayInt,
    NDArrayNumber,
    NDArrayUI64,
)
from vis4d.data.const import CommonKeys as K

from .base import Transform


def fnv_hash_vec(arr: NDArrayInt) -> NDArrayUI64:
    """FNV64-1A has function for 2d array shape (N, 3)."""
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr_64 = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(
        arr.shape[0], dtype=np.uint64
    )
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr_64[:, j])
    return hashed_arr


class VoxelMapping(TypedDict):
    """Voxel mapping parameters.

    Attributes:
        point_idx (NDArrayInt): Point indices. Shape (n_voxels,).
        voxel_idx (NDArrayInt): Voxel indices for each point.
            Shape (n_points,).
    """

    point_idx: NDArrayInt
    voxel_idx: NDArrayInt


@Transform(
    (K.points3d,),
    "transforms.voxel_mapping",
)
class GenVoxelMapping:
    """Extracts VoxelMapping from a given pointcloud.

    The VoxelMappings can be used to extract voxelized pointclouds from
    the original pointcloud using the Voixelize[Points, Colors, ....]]
    transforms.
    """

    def __init__(
        self,
        voxel_size: float = 0.05,
        random_downsample: bool = True,
        max_voxels: int | None = None,
        shuffle: bool = False,
    ) -> None:
        """Creates an instance of the class.

        Args:
            voxel_size (float): Voxel size (in meters). Default: 0.05
            random_downsample (bool): Whether to randomly downsample.
                If True, the center point for downsampling is randomly
                selected. Default: True If max_voxels is None, this is ignored.
            max_voxels (int | None): Maximum number of voxels. Default: None
            shuffle (bool): Whether to shuffle the indices. Default: False
        """
        self.voxel_size = voxel_size
        self.random_downsample = random_downsample
        self.max_voxels = max_voxels
        self.shuffle = shuffle

    def __call__(self, data_list: list[NDArrayNumber]) -> list[VoxelMapping]:
        """Samples num_pts from the first dim of the provided data tensor.

        If num_pts > data.shape[0], the indices will be upsampled with
        replacement. If num_pts < data.shape[0], the indices will be sampled
        without replacement.

        Args:
            data_list (list[NDArrayNumber]): Data from which to sample indices.

        Returns:
            list[VoxelMapping]: List of sampled voxel information.

        Raises:
            ValueError: If data is empty.
        """
        return_data = []
        # FIXME, if downsampling is used, returned voxel_idx is not correct.
        # Currently unused.

        for data in data_list:
            discrete_coord = np.floor(
                (data - data.min(0)) / np.array(self.voxel_size)
            )
            key = fnv_hash_vec(discrete_coord)
            idx_sort = np.argsort(key)
            key_sort = key[idx_sort]

            _, voxel_idx, count = np.unique(
                key_sort, return_counts=True, return_inverse=True
            )

            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_sort = idx_sort[idx_select]

            if self.max_voxels is not None:
                n_voxels = len(idx_sort)
                if n_voxels > self.max_voxels:
                    # Downsample based on distance to the center
                    if self.random_downsample:
                        center_idx = np.random.randint(n_voxels)
                    else:
                        center_idx = n_voxels // 2
                    crop_idx = np.argsort(
                        np.sum(
                            np.square(
                                data[idx_sort] - data[idx_sort[center_idx]]
                            ),
                            1,
                        )
                    )[: self.max_voxels]

                    idx_sort = idx_sort[crop_idx]
                else:
                    query_inds = np.arange(n_voxels)
                    padding_choice = np.random.choice(
                        n_voxels, self.max_voxels - n_voxels
                    )
                    crop_idx = np.hstack(
                        [query_inds, query_inds[padding_choice]]
                    )
                    idx_sort = idx_sort[crop_idx]

            if self.shuffle:
                np.random.shuffle(idx_sort)

            return_data.append(
                VoxelMapping(point_idx=idx_sort, voxel_idx=voxel_idx)
            )
        return return_data


@Transform(
    (K.points3d, "transforms.voxel_mapping"),
    K.points3d,
)
class VoxelizePoints:
    """Voxelize points.

    Requires to generate VoxelMapping first.
    See GenVoxelMapping.
    """

    def __call__(
        self, points3d: list[NDArrayFloat], voxel_mapping: list[VoxelMapping]
    ) -> list[NDArrayNumber]:
        """Samples num_pts from the first dim of the provided data tensor."""
        ret_data = []
        for points, voxel_params in zip(points3d, voxel_mapping):
            ret_data.append(points[voxel_params["point_idx"]])
        return ret_data


@Transform((K.semantics3d, "transforms.voxel_mapping"), K.semantics3d)
class VoxelizeSemantics(VoxelizePoints):
    """Voxelize Semantics.

    Requires to generate VoxelMapping first.
    See GenVoxelMapping.
    """


@Transform((K.instances3d, "transforms.voxel_mapping"), K.instances3d)
class VoxelizeInstances(VoxelizePoints):
    """Voxelize Instances.

    Requires to generate VoxelMapping first.
    See GenVoxelMapping.
    """


@Transform((K.colors3d, "transforms.voxel_mapping"), K.colors3d)
class VoxelizeColors(VoxelizePoints):
    """Voxelize Colors.

    Requires to generate VoxelMapping first.
    See GenVoxelMapping.
    """
