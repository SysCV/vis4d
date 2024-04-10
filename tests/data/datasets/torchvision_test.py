"""Test for Torchvision dataset wrapper."""

import os
import shutil
import tempfile
import unittest

from torchvision.datasets.mnist import MNIST

from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets import TorchvisionClassificationDataset


class TorchvisionClassifcationTest(unittest.TestCase):
    """Test torchvision classification datsets."""

    def setUp(self) -> None:
        """Creates a tmp directory and loads input data."""
        self.test_dir = tempfile.mkdtemp()
        self.train_ds = TorchvisionClassificationDataset(
            MNIST(
                os.path.join(self.test_dir, "mnist"), train=True, download=True
            )
        )
        self.test_ds = TorchvisionClassificationDataset(
            MNIST(os.path.join(self.test_dir, "mnist"), train=False)
        )

    def tearDown(self) -> None:
        """Removes the tmp directory."""
        shutil.rmtree(self.test_dir)

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.train_ds), 60000)
        self.assertEqual(len(self.test_ds), 10000)

    def test_data_shape(self) -> None:
        """Test if data shape is correct."""
        data = next(iter(self.train_ds))
        self.assertTrue(data[K.images].shape == (1, 28, 28, 1))
        self.assertTrue(data[K.categories].shape == (1,))

        data = next(iter(self.test_ds))
        self.assertTrue(data[K.images].shape == (1, 28, 28, 1))
        self.assertTrue(data[K.categories].shape == (1,))
