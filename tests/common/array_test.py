# mypy: disable-error-code=call-overload
"""Testcases for Vis4D array conversion methods."""

import unittest

import numpy as np
import torch

from vis4d.common.array import array_to_numpy


class TestConvertToArray(unittest.TestCase):
    """Test cases for array conversion ops."""

    def test_none(self) -> None:
        """Test that None is returned as None."""
        self.assertEqual(array_to_numpy(None), None)

    def test_array(self) -> None:
        """Test that numpy arrays are returned as is."""
        data = np.random.rand(2, 3)
        self.assertTrue(
            np.allclose(data, array_to_numpy(data, dtype=np.float64))
        )

    def test_iterable(self) -> None:
        """Test that iterables are converted to numpy arrays."""
        data = [[1, 2, 3], [2, 3, 4]]
        self.assertTrue(np.allclose(np.asarray(data), array_to_numpy(data)))

    def test_torch(self) -> None:
        """Test that torch tensors are converted to numpy arrays."""
        data = np.asarray([[1, 2, 3], [2, 3, 4]])
        self.assertTrue(
            np.allclose(data, array_to_numpy(torch.from_numpy(data)))
        )

    def test_dim_shaping(self) -> None:
        """Test that the number of dimensions is correct."""
        data = np.random.rand(2, 3)

        # Check expanding works
        self.assertEqual(
            array_to_numpy(data.copy(), dtype=np.float32).shape, (2, 3)
        )
        self.assertEqual(array_to_numpy(data.copy(), 3).shape, (1, 2, 3))
        self.assertEqual(array_to_numpy(data.copy(), 4).shape, (1, 1, 2, 3))

        data = data.reshape((1, 2, 3, 1))
        # Make sure we remove from the left as default
        self.assertEqual(array_to_numpy(data.copy(), 3).shape, (2, 3, 1))

        # And the right if we can not remove anything from the left anymore
        self.assertEqual(array_to_numpy(data.copy(), 2).shape, (2, 3))
