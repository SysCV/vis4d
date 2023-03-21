"""Point sampling transforms testing class."""
from __future__ import annotations

import copy
import unittest

import numpy as np
import pytest
import torch

from vis4d.data.const import CommonKeys
from vis4d.data.transforms.point_sampling import (
    GenerateSamplingIndices,
    SamplePoints,
    sample_from_block,
    sample_points_block_full_coverage,
)
from vis4d.data.typing import DictData


class TestSampleFromBlock(unittest.TestCase):
    """Tests sampling in a block based fashion."""

    data_in_unit_square = torch.rand(100, 3).sort(dim=0).values
    data_outside_unit_square = torch.rand(100, 3).sort(dim=0).values + 10
    n_pts_to_sample = 100
    data: dict[str, torch.Tensor] = {}
    original_data: dict[str, torch.Tensor] = {}

    @pytest.fixture(autouse=True)
    def initdata(self) -> None:
        """Loads dummy data."""
        self.data = {
            CommonKeys.points3d: torch.cat(
                [self.data_in_unit_square, self.data_outside_unit_square]
            ),
            CommonKeys.colors3d: torch.rand(200, 3),
            CommonKeys.semantics3d: torch.randint(10, (200, 1)),
        }
        self.original_data = copy.deepcopy(self.data)

    def test_block_sampling(self) -> None:
        """Tests the functional."""
        # Should return the full block
        data_to_sample_from = self.data_in_unit_square
        _, sampled_idxs = sample_from_block(
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
        _, sampled_idxs = sample_from_block(
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
        _, sampled_idxs = sample_from_block(
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

    def test_full_scale_block_sampling(self) -> None:
        """Tests if all points are sampled when using full coverage."""
        # pylint: disable=unexpected-keyword-arg
        sampler = sample_points_block_full_coverage(
            min_pts_per_block=1,
            n_pts_per_block=200,
            in_keys=(
                CommonKeys.points3d,
                CommonKeys.semantics3d,
                CommonKeys.colors3d,
            ),
            out_keys=(
                CommonKeys.points3d,
                CommonKeys.semantics3d,
                CommonKeys.colors3d,
            ),
        )

        data_sampled = sampler(self.data)
        for key in (
            CommonKeys.points3d,
            CommonKeys.semantics3d,
            CommonKeys.colors3d,
        ):
            self.assertTrue(
                torch.all(
                    data_sampled[key].unique(dim=0)
                    == self.original_data[key].unique(dim=0)
                ).item()
            )


class RandomPointSamplingTest(unittest.TestCase):
    """Test point sampling."""

    def test_sample_less_pts(self) -> None:
        """Test if sampling works when sampling less pts than in the scene."""

        data: DictData = dict(points3d=np.random.rand(100, 3))
        tr1 = GenerateSamplingIndices(num_idxs=10)
        tr2 = SamplePoints()
        with_idxs = tr1.apply_to_data(data)
        sampled_points = tr2.apply_to_data(with_idxs)
        self.assertEqual(sampled_points["points3d"].shape[0], 10)

    def test_sample_more_pts(self) -> None:
        """Tests if sampling works with more points.

        It uses more points than given in the scene.
        """

        data: DictData = dict(points3d=np.random.rand(100, 3))
        tr1 = GenerateSamplingIndices(num_idxs=1000)
        tr2 = SamplePoints()
        with_idxs = tr1.apply_to_data(data)
        sampled_points = tr2.apply_to_data(with_idxs)
        self.assertEqual(sampled_points["points3d"].shape[0], 1000)
