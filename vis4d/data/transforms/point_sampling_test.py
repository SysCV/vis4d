"""point sampling augmentation testing class."""
import copy
import unittest

import pytest
import torch
from numpy import block

from ..datasets.base import DataKeys
from .point_sampling import (
    FullCoverageBlockSampler,
    RandomBlockPointSampler,
    RandomPointSampler,
    sample_from_block,
)


class TestSampleFromBlock(unittest.TestCase):
    """Tests sampling in a block based fashion."""

    data_in_unit_square = torch.rand(100, 3).sort(dim=0).values
    data_outside_unit_square = torch.rand(100, 3).sort(dim=0).values + 10
    n_pts_to_sample = 100

    @pytest.fixture(autouse=True)
    def initdata(self):
        """Loads dummy data."""
        self.data = {
            DataKeys.points3d: torch.cat(
                [self.data_in_unit_square, self.data_outside_unit_square]
            ),
            DataKeys.colors3d: torch.rand(200, 3),
            DataKeys.semantics3d: torch.randint(10, (200, 1)),
        }
        self.original_data = copy.deepcopy(self.data)

    def test_block_sampling(self):
        """Tests the functional."""
        # Should return the full block
        data_to_sample_from = self.data_in_unit_square
        n_pts, sampled_idxs = sample_from_block(
            self.n_pts_to_sample,
            data_to_sample_from,
            center_xyz=torch.tensor([0.5, 0.5, 0.5]),
            block_size=torch.tensor([1, 1, 1]),
        )
        self.assertTrue(
            torch.all(
                data_to_sample_from[sampled_idxs].sort(dim=0).values
                == self.data_in_unit_square
            ).item()
        )

        # Should only sample from the first block
        data_to_sample_from = torch.cat(
            [self.data_in_unit_square, self.data_outside_unit_square]
        )
        n_pts, sampled_idxs = sample_from_block(
            self.n_pts_to_sample,
            data_to_sample_from,
            center_xyz=torch.tensor([0.5, 0.5, 0.5]),
            block_size=torch.tensor([1, 1, 1]),
        )
        self.assertTrue(
            torch.all(
                data_to_sample_from[sampled_idxs].sort(dim=0).values
                == self.data_in_unit_square
            ).item()
        )

        # Should only sample from the second block
        data_to_sample_from = torch.cat(
            [self.data_in_unit_square, self.data_outside_unit_square]
        )
        n_pts, sampled_idxs = sample_from_block(
            self.n_pts_to_sample,
            data_to_sample_from,
            center_xyz=torch.tensor([10.5, 10.5, 10.5]),
            block_size=torch.tensor([1, 1, 1]),
        )
        self.assertTrue(
            torch.all(
                data_to_sample_from[sampled_idxs].sort(dim=0).values
                == self.data_outside_unit_square
            ).item()
        )

    def test_sampler(self):
        """Test the Class implementation of the sampling functional."""
        sampler = RandomBlockPointSampler(
            n_pts=500,
            min_pts=10,
            coord_key=DataKeys.points3d,
            in_keys=(
                DataKeys.points3d,
                DataKeys.semantics3d,
                DataKeys.colors3d,
            ),
        )
        data_sampled = sampler(self.data, None)
        self.assertEqual(data_sampled[DataKeys.points3d].size(0), 500)
        self.assertEqual(data_sampled[DataKeys.semantics3d].size(0), 500)
        self.assertEqual(data_sampled[DataKeys.colors3d].size(0), 500)

    def test_full_scale_block_sampling(self):
        """Tests if all points are sampled when using full coverage and enough points"""
        sampler = FullCoverageBlockSampler(
            min_pts_per_block=1,
            n_pts_per_block=200,
            in_keys=(
                DataKeys.points3d,
                DataKeys.colors3d,
                DataKeys.semantics3d,
            ),
        )

        data_sampled = sampler(
            self.data, sampler.generate_parameters(self.data)
        )
        for key in (
            DataKeys.points3d,
            DataKeys.semantics3d,
            DataKeys.colors3d,
        ):
            self.assertTrue(
                torch.all(
                    data_sampled[key].unique(dim=0)
                    == self.original_data[key].unique(dim=0)
                ).item()
            )


class RandomPointSamplingTest(unittest.TestCase):
    """Test point sampling."""

    n_scene_pts = 1000

    @pytest.fixture(autouse=True)
    def initdata(self):
        """Loads dummy data."""
        self.data = {
            DataKeys.points3d: torch.rand(self.n_scene_pts, 3),
            DataKeys.colors3d: torch.rand(self.n_scene_pts, 3),
            DataKeys.semantics3d: torch.rand(self.n_scene_pts, 1),
        }

    def test_sample_less_pts(self):
        """Test if sampling works when sampling less points than given in
        the scene."""
        sampler = RandomPointSampler(
            n_pts=100, in_keys=(DataKeys.points3d, DataKeys.semantics3d)
        )
        data_sampled = sampler(self.data, None)
        self.assertEqual(data_sampled[DataKeys.points3d].size(0), 100)
        self.assertEqual(data_sampled[DataKeys.semantics3d].size(0), 100)
        self.assertEqual(
            data_sampled[DataKeys.colors3d].size(0), self.n_scene_pts
        )

    def test_sample_more_pts(self):
        """Tests if sampling works when sampling more points tha given in
        the scene"""

        sampler = RandomPointSampler(
            n_pts=10000,
            in_keys=(
                DataKeys.points3d,
                DataKeys.semantics3d,
                DataKeys.colors3d,
            ),
        )
        data_sampled = sampler(self.data, None)
        self.assertEqual(data_sampled[DataKeys.points3d].size(0), 10000)
        self.assertEqual(data_sampled[DataKeys.semantics3d].size(0), 10000)
        self.assertEqual(data_sampled[DataKeys.colors3d].size(0), 10000)
