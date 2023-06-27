# pylint: disable=no-member,unexpected-keyword-arg,use-dict-literal
"""Point sampling transforms testing class."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms import compose
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

    def test_block_sampling(self) -> None:
        """Tests the functor."""
        # Should return the full block
        data_to_sample_from = {K.points3d: self.data_in_unit_square}

        mask_gen = GenerateBlockSamplingIndices(
            self.n_pts_to_sample,
            block_dimensions=(1, 1, 1),
            center_point=(0.5, 0.5, 0.5),
        )

        sampler = SamplePoints()

        tr1 = compose([mask_gen, sampler])

        data_sampled = tr1([data_to_sample_from])[0]

        self.assertTrue(
            np.all(
                np.sort(data_sampled[K.points3d], axis=0)
                == self.data_in_unit_square
            )
        )

        # Should only sample from the first block
        data_to_sample_from = {
            K.points3d: np.concatenate(
                [self.data_in_unit_square, self.data_outside_unit_square]
            )
        }
        mask_gen = GenerateBlockSamplingIndices(
            self.n_pts_to_sample,
            block_dimensions=(1, 1, 1),
            center_point=(0.5, 0.5, 0.5),
        )

        tr2 = compose([mask_gen, sampler])

        data_sampled = tr2([data_to_sample_from])[0]
        self.assertTrue(
            np.all(
                np.sort(data_sampled[K.points3d], axis=0)
                == self.data_in_unit_square
            )
        )

        # Should only sample from the second block
        data_to_sample_from = {
            K.points3d: np.concatenate(
                [self.data_in_unit_square, self.data_outside_unit_square]
            )
        }
        mask_gen = GenerateBlockSamplingIndices(
            self.n_pts_to_sample,
            block_dimensions=(1, 1, 1),
            center_point=(10.5, 10.5, 10.5),
        )

        tr3 = compose([mask_gen, sampler])

        data_sampled = tr3([data_to_sample_from])[0]
        self.assertTrue(
            np.all(
                np.sort(data_sampled[K.points3d], axis=0)
                == self.data_outside_unit_square
            )
        )

    def test_full_scale_block_sampling(self) -> None:
        """Tests if all points are sampled when using full coverage."""
        data = {
            K.points3d: np.concatenate(
                [
                    self.data_in_unit_square,
                    self.data_in_unit_square,
                ]
            )
        }

        mask_gen = GenFullCovBlockSamplingIndices(
            block_dimensions=(1, 1, 1),
            min_pts=1,
            num_pts=200,
        )
        sampler = SamplePoints()

        transform = compose([mask_gen, sampler])

        data_sampled = transform([copy.deepcopy(data)])[0]

        self.assertTrue(
            np.all(
                np.unique(data_sampled[K.points3d].reshape(-1, 3), axis=0)
                == np.unique(data[K.points3d], axis=00)
            )
        )


class RandomPointSamplingTest(unittest.TestCase):
    """Test point sampling."""

    def test_sample_less_pts(self) -> None:
        """Test if sampling works when sampling less pts than in the scene."""
        data: DictData = dict(points3d=np.random.rand(100, 3))
        tr1 = GenerateSamplingIndices(num_pts=10)
        tr2 = SamplePoints()

        transform = compose([tr1, tr2])
        sampled_points = transform([data])[0]

        self.assertEqual(sampled_points["points3d"].shape[0], 10)

    def test_sample_more_pts(self) -> None:
        """Tests if sampling works with more points.

        It uses more points than given in the scene.
        """
        data: DictData = dict(points3d=np.random.rand(100, 3))
        tr1 = GenerateSamplingIndices(num_pts=1000)
        tr2 = SamplePoints()

        transform = compose([tr1, tr2])
        sampled_points = transform([data])[0]

        self.assertEqual(sampled_points["points3d"].shape[0], 1000)
