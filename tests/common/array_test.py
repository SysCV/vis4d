"""Testcases for Vis4D array conversion methods."""

import unittest

import numpy as np
import torch

from vis4d.common.array import array_to_numpy, arrays_to_numpy


class TestConvertToArray(unittest.TestCase):
    """Test cases for array conversion ops."""

    def test_none(self) -> None:
        self.assertEqual(array_to_numpy(None), None)

    def test_array(self) -> None:
        data = np.random.rand(2, 3)
        self.assertFalse((data - array_to_numpy(data)).any())

    def test_iterable(self) -> None:
        data = [[1, 2, 3], [2, 3, 4]]
        self.assertFalse((np.asarray(data) - array_to_numpy(data)).any())

    def test_torch(self) -> None:
        data = np.asarray([[1, 2, 3], [2, 3, 4]])
        self.assertFalse((data - array_to_numpy(torch.from_numpy(data))).any())

    def test_dim_shaping(self) -> None:
        data = np.random.rand(2, 3)

        # Check expanding works
        self.assertEqual(array_to_numpy(data.copy()).shape, (2, 3))
        self.assertEqual(array_to_numpy(data.copy(), 3).shape, (1, 2, 3))
        self.assertEqual(array_to_numpy(data.copy(), 4).shape, (1, 1, 2, 3))

        data = data.reshape((1, 2, 3, 1))
        # Make sure we remove from the left as default
        self.assertEqual(array_to_numpy(data.copy(), 3).shape, (2, 3, 1))

        # And the right if we can not remove anything from the left anymore
        self.assertEqual(array_to_numpy(data.copy(), 2).shape, (2, 3))

    def test_array_to_numpys_multiple(self) -> None:
        out = arrays_to_numpy(
            np.random.rand(2, 3), np.random.rand(1, 1, 2, 3), n_dims=3
        )
        for arr in out:
            self.assertEqual(arr.shape, (1, 2, 3))
