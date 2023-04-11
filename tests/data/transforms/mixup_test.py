# tests/transforms/test_mixup.py
"""Tests for Mixup."""

import unittest

import numpy as np

from vis4d.common.util import set_random_seed
from vis4d.data.transforms.mixup import Mixup


class TestMixup(unittest.TestCase):
    """Tests for Mixup."""

    def test_batch_mixup(self):
        """Test batch mixup."""
        set_random_seed(0, deterministic=True)

        mixup = Mixup(alpha=1.0, num_classes=2)
        images = [
            np.ones((32, 32, 3)).astype(np.float32),
            np.zeros((32, 32, 3)).astype(np.float32),
        ]
        categories = [np.array([0.0, 1.0]), np.array([1.0, 0.0])]
        images, categories = mixup(images, categories)

        self.assertEqual(len(images), 2)
        self.assertEqual(len(categories), 2)
        self.assertEqual(categories[0].shape, (2,))
        self.assertEqual(categories[1].shape, (2,))
        self.assertEqual(images[0].shape, (32, 32, 3))
        self.assertEqual(images[1].shape, (32, 32, 3))

        self.assertAlmostEqual(images[0][0, 0, 0], 0.5509, places=3)
        self.assertAlmostEqual(images[1][0, 0, 0], 0.2474, places=3)

        self.assertAlmostEqual(categories[0][0], 0.4267, places=3)
        self.assertAlmostEqual(categories[0][1], 0.5233, places=3)
        self.assertAlmostEqual(categories[1][0], 0.5233, places=3)
        self.assertAlmostEqual(categories[1][1], 0.4267, places=3)
