"""Point sampling transforms testing class."""
from __future__ import annotations

import copy
import unittest

import numpy as np
import pytest

from vis4d.data.const import CommonKeys
from vis4d.data.transforms.point_sampling import (
    GenerateBlockSamplingIndices,
    GenerateSamplingIndices,
    GenFullCovBlockSamplingIndices,
    SamplePoints,
)
from vis4d.data.typing import DictData


class TestSampleFromBlock(unittest.TestCase):
    """Tests sampling in a block based fashion."""

    data_in_unit_square = np.sort(np.random.rand(100, 3), axis=0)
    data_outside_unit_square = np.sort(np.random.rand(100, 3), axis=0) + 10
    n_pts_to_sample = 100
    data: dict[str, np.ndarray] = {}
    original_data: dict[str, np.ndarray] = {}

    @pytest.fixture(autouse=True)
    def initdata(self) -> None:
        """Loads dummy data."""
        self.data = {
            CommonKeys.points3d: np.concatenate(
                [self.data_in_unit_square, self.data_outside_unit_square]
            ),
        }
        self.original_data = copy.deepcopy(self.data)

    def test_block_sampling(self) -> None:
        """Tests the functor."""
        # Should return the full block
        data_to_sample_from = {CommonKeys.points3d: self.data_in_unit_square}

        mask_gen = GenerateBlockSamplingIndices(
            self.n_pts_to_sample,
            block_dimensions=(1, 1, 1),
            center_point=(0.5, 0.5, 0.5),
        )

        sampler = SamplePoints()
        data_sampled = sampler.apply_to_data(
            mask_gen.apply_to_data(data_to_sample_from)
        )

        self.assertTrue(
            np.all(
                np.sort(data_sampled[CommonKeys.points3d], axis=0)
                == self.data_in_unit_square
            )
        )

        # Should only sample from the first block
        data_to_sample_from = {
            CommonKeys.points3d: np.concatenate(
                [self.data_in_unit_square, self.data_outside_unit_square]
            )
        }
        mask_gen = GenerateBlockSamplingIndices(
            self.n_pts_to_sample,
            block_dimensions=(1, 1, 1),
            center_point=(0.5, 0.5, 0.5),
        )

        data_sampled = sampler.apply_to_data(
            mask_gen.apply_to_data(data_to_sample_from)
        )
        self.assertTrue(
            np.all(
                np.sort(data_sampled[CommonKeys.points3d], axis=0)
                == self.data_in_unit_square
            )
        )
        # Should only sample from the second block
        data_to_sample_from = {
            CommonKeys.points3d: np.concatenate(
                [self.data_in_unit_square, self.data_outside_unit_square]
            )
        }
        mask_gen = GenerateBlockSamplingIndices(
            self.n_pts_to_sample,
            block_dimensions=(1, 1, 1),
            center_point=(10.5, 10.5, 10.5),
        )

        data_sampled = sampler.apply_to_data(
            mask_gen.apply_to_data(data_to_sample_from)
        )
        self.assertTrue(
            np.all(
                np.sort(data_sampled[CommonKeys.points3d], axis=0)
                == self.data_outside_unit_square
            )
        )

    def test_full_scale_block_sampling(self) -> None:
        """Tests if all points are sampled when using full coverage."""
        mask_gen = GenFullCovBlockSamplingIndices(
            block_dimensions=(1, 1, 1),
            min_pts=1,
            num_pts=200,
        )

        sampler = SamplePoints()
        data_sampled = sampler.apply_to_data(mask_gen.apply_to_data(self.data))
        self.assertTrue(
            np.all(
                np.unique(
                    data_sampled[CommonKeys.points3d].reshape(-1, 3), axis=0
                )
                == np.unique(self.original_data[CommonKeys.points3d], axis=00)
            )
        )


class RandomPointSamplingTest(unittest.TestCase):
    """Test point sampling."""

    def test_sample_less_pts(self) -> None:
        """Test if sampling works when sampling less pts than in the scene."""
        data: DictData = dict(points3d=np.random.rand(100, 3))
        tr1 = GenerateSamplingIndices(num_pts=10)
        tr2 = SamplePoints()
        with_idxs = tr1.apply_to_data(data)
        sampled_points = tr2.apply_to_data(with_idxs)
        self.assertEqual(sampled_points["points3d"].shape[0], 10)

    def test_sample_more_pts(self) -> None:
        """Tests if sampling works with more points.

        It uses more points than given in the scene.
        """
        data: DictData = dict(points3d=np.random.rand(100, 3))
        tr1 = GenerateSamplingIndices(num_pts=1000)
        tr2 = SamplePoints()
        with_idxs = tr1.apply_to_data(data)
        sampled_points = tr2.apply_to_data(with_idxs)
        self.assertEqual(sampled_points["points3d"].shape[0], 1000)
