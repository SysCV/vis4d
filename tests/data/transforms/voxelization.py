# pylint: disable=no-member,unexpected-keyword-arg,use-dict-literal
"""Point-voxelization transformation testing class."""
import pickle
import unittest

import numpy as np

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms import compose
from vis4d.data.transforms.voxelize import (
    GenVoxelMapping,
    VoxelizeColors,
    VoxelizeInstances,
    VoxelizePoints,
    VoxelizeSemantics,
)


class TestVoxelization(unittest.TestCase):
    """Tests sampling in a block based fashion."""

    voxel_size = 0.04
    voxel_max = 24000
    presample = False
    shuffle = False

    def test_voxelization(self) -> None:
        """Test if voxelization produces the correct output."""
        with open(
            get_test_data("transforms/voxelize/voxelize_tf.pkl"), "rb"
        ) as f:
            test_data = pickle.load(f)
        data_in = test_data["input"]
        expected_out = test_data["voxelized"]

        data = {
            K.points3d: data_in["points"],
            K.colors3d: data_in["features"],
            K.semantics3d: data_in["labels"],
            K.instances3d: data_in["labels"],
        }

        mask_gen = GenVoxelMapping(
            self.voxel_size,
            random_downsample=True,
            max_voxels=self.voxel_max,
            shuffle=self.shuffle,
        )
        tr1 = compose(
            [
                mask_gen,
                VoxelizePoints(),
                VoxelizeColors(),
                VoxelizeSemantics(),
                VoxelizeInstances(),
            ]
        )
        np.random.seed(0)
        data_sampled = tr1([data])[0]
        self.assertTrue(
            np.allclose(data_sampled[K.points3d], expected_out["points"])
        )
        self.assertTrue(
            np.allclose(data_sampled[K.colors3d], expected_out["features"])
        )
        self.assertTrue(
            np.allclose(data_sampled[K.semantics3d], expected_out["labels"])
        )
        self.assertTrue(
            np.allclose(data_sampled[K.instances3d], expected_out["labels"])
        )
