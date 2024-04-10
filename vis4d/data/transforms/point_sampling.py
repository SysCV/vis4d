"""Contains different Sampling Trasnforms for pointclouds."""

from __future__ import annotations

import numpy as np

from vis4d.common.typing import NDArrayInt, NDArrayNumber
from vis4d.data.const import CommonKeys as K

from .base import Transform


@Transform(K.points3d, "transforms.sampling_idxs")
class GenerateSamplingIndices:
    """Samples num_pts from the first dim of the provided data tensor.

    If num_pts > data.shape[0], the indices will be upsampled with
    replacement. If num_pts < data.shape[0], the indices will be sampled
    without replacement.
    """

    def __init__(self, num_pts: int) -> None:
        """Creates an instance of the class.

        Args:
            num_pts (int): Number of indices to sample
        """
        self.num_pts = num_pts

    def __call__(self, data_list: list[NDArrayNumber]) -> list[NDArrayInt]:
        """Samples num_pts from the first dim of the provided data tensor.

        If num_pts > data.shape[0], the indices will be upsampled with
        replacement. If num_pts < data.shape[0], the indices will be sampled
        without replacement.

        Args:
            data_list (list[NDArrayNumber]): Data from which to sample indices.

        Returns:
            list[NDArrayInt]: List of indices.

        Raises:
            ValueError: If data is empty.
        """
        data = data_list[0]

        if len(data) == 0:
            raise ValueError("Data sample was empty!")

        if self.num_pts > len(data):
            return [
                np.concatenate(
                    [
                        np.arange(len(data)),
                        np.random.randint(
                            0, len(data), self.num_pts - len(data)
                        ),
                    ]
                )
            ] * len(data_list)
        return [
            np.random.choice(len(data), self.num_pts, replace=False)
        ] * len(data_list)


@Transform(K.points3d, "transforms.sampling_idxs")
class GenerateBlockSamplingIndices:
    """Samples num_pts from the first dim of the provided data tensor.

    Makes sure that the sampled points are within a block of size block_size
    centered around center_xyz. If num_pts > data.shape[0], the indices will
    be upsampled with replacement. If num_pts < data.shape[0], the indices
    will be sampled without replacement.
    """

    def __init__(
        self,
        num_pts: int,
        block_dimensions: tuple[float, float, float],
        center_point: tuple[float, float, float] | None = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            num_pts (int): Number of indices to sample
            block_dimensions (tuple[float, float, float]): Dimensions of the
                block in x,y,z
            center_point (tuple[float, float, float] | None): Center point of
                the block in x,y,z. If None, the center will be sampled
                randomly.
        """
        self.block_dimensions = np.asarray(block_dimensions)
        self.center_point = (
            np.asarray(center_point) if center_point is not None else None
        )

        self._idx_sampler = GenerateSamplingIndices(num_pts)

    def __call__(self, data_list: list[NDArrayNumber]) -> list[NDArrayInt]:
        """Samples num_pts from the first dim of the provided data tensor."""
        data = data_list[0]

        if self.center_point is None:
            center_point = data[np.random.choice(len(data), 1)]
        else:
            center_point = self.center_point

        max_box = center_point + self.block_dimensions / 2.0
        min_box = center_point - self.block_dimensions / 2.0

        box_mask = np.logical_and(
            np.all(data >= min_box, axis=1),
            np.all(data <= max_box, axis=1),
        )
        if box_mask.sum().item() == 0:  # No valid data sample found!
            return [np.array([], dtype=np.int32)] * len(data_list)

        idxs = self._idx_sampler([data[box_mask, ...]])[0]

        masked_idxs = np.arange(data.shape[0])[box_mask]
        selected_idxs_global = masked_idxs[idxs]
        return [selected_idxs_global] * len(data_list)


@Transform(K.points3d, "transforms.sampling_idxs")
class GenFullCovBlockSamplingIndices:
    """Subsamples the pointcloud using blocks of a given size."""

    def __init__(
        self,
        num_pts: int,
        block_dimensions: tuple[float, float, float],
        min_pts: int = 32,
    ) -> None:
        """Creates an instance of the class.

        Args:
            num_pts (int): Number of points to sample for each block
            block_dimensions (tuple[float, float, float]): Dimensions of the
                block in x,y,z
            min_pts (int): Minimum number of points in a block to be considered
                valid
        """
        self.num_pts = num_pts
        self.min_pts = min_pts
        self.block_dimensions = np.asarray(block_dimensions)
        self._idx_sampler = GenerateBlockSamplingIndices(
            num_pts=self.num_pts,
            block_dimensions=block_dimensions,
        )

    def __call__(
        self, coordinates_list: list[NDArrayNumber]
    ) -> list[NDArrayInt]:
        """Subsamples the pointcloud using blocks of a given size."""
        coordinates = coordinates_list[0]

        # Get bounding box for sampling
        coord_min, coord_max = (
            np.min(coordinates, axis=0),
            np.max(coordinates, axis=0),
        )
        sampled_idxs = []
        hwl = coord_max - coord_min
        num_blocks = np.ceil(hwl / self.block_dimensions).astype(np.int32)

        for idx_x in range(num_blocks[0]):
            for idx_y in range(num_blocks[1]):
                for idx_z in range(num_blocks[2]):
                    center_pt = (
                        coord_min
                        + np.array([idx_x, idx_y, idx_z])
                        * self.block_dimensions
                        + self.block_dimensions / 2.0
                    )

                    self._idx_sampler.center_point = center_pt
                    selected_idxs = self._idx_sampler([coordinates])[0]
                    if selected_idxs.sum() >= self.min_pts:
                        sampled_idxs.append(selected_idxs)
        return [np.stack(sampled_idxs)] * len(coordinates_list)  # type: ignore


@Transform([K.points3d, "transforms.sampling_idxs"], K.points3d)
class SamplePoints:
    """Subsamples points randomly.

    Samples 'num_pts' randomly from the provided data tensors using the
    provided sampling indices.

    This transform is used to sample points from a pointcloud. The indices
    are generated by the GenerateSamplingIndices transform.

    """

    def __call__(
        self,
        data_list: list[NDArrayNumber],
        selected_idxs_list: list[NDArrayInt],
    ) -> list[NDArrayNumber]:
        """Returns data[selected_idxs].

        If the provided indices have two dimension (i.e n_masks, 64), then
        this operation indices the data n_masks times and returns an array
        """
        for i, (data, selected_idxs) in enumerate(
            zip(data_list, selected_idxs_list)
        ):
            assert selected_idxs.ndim <= 2, "Indices must be 1D or 2D"
            if selected_idxs.ndim == 2:
                data_list[i] = np.stack(
                    [data[idxs, ...] for idxs in selected_idxs]
                )
            else:
                data_list[i] = data[selected_idxs, ...]
        return data_list


@Transform([K.colors3d, "transforms.sampling_idxs"], K.colors3d)
class SampleColors(SamplePoints):
    """Subsamples colors randomly.

    Samples 'num_pts' randomly from the provided data tensors using the
    provided sampling indices.

    This transform is used to sample colors from a pointcloud. The indices
    are generated by the GenerateSamplingIndices transform.
    """


@Transform([K.semantics3d, "transforms.sampling_idxs"], K.semantics3d)
class SampleSemantics(SamplePoints):
    """Subsamples semantics randomly.

    Samples 'num_pts' randomly from the provided data tensors using the
    provided sampling indices.

    This transform is used to sample semantics from a pointcloud. The indices
    are generated by the GenerateSamplingIndices transform.
    """


@Transform([K.instances3d, "transforms.sampling_idxs"], K.instances3d)
class SampleInstances(SamplePoints):
    """Subsamples instances randomly.

    Samples 'num_pts' randomly from the provided data tensors using the
    provided sampling indices.

    This transform is used to sample instances from a pointcloud. The indices
    are generated by the GenerateSamplingIndices transform.
    """
