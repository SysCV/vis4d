# tests/transforms/test_mixup.py
"""Tests for Mixup."""

import numpy as np
import unittest

from vis4d.common.util import set_random_seed
from vis4d.data.transforms.mixup import Mixup


class TestMixup(unittest.TestCase):
    """Tests for Mixup."""

    def test_batch_mixup(self):
        """Test batch mixup."""
        set_random_seed(0, deterministic=True)

        mixup = Mixup(probability=1.0, alpha=1.0, num_classes=2)
        images = [np.ones((32, 32, 3)), np.zeros((32, 32, 3))]
        categories = [np.array(1), np.array(0)]
        images, smooth_categories = mixup(images, categories)

        self.assertEqual(len(images), 2)
        self.assertEqual(len(smooth_categories), 2)
        self.assertEqual(smooth_categories[0].shape, (2,))
        self.assertEqual(smooth_categories[1].shape, (2,))
        self.assertEqual(images[0].shape, (32, 32, 3))
        self.assertEqual(images[1].shape, (32, 32, 3))

        self.assertAlmostEqual(images[0][0, 0, 0], 0.5623, places=3)
        self.assertAlmostEqual(images[1][0, 0, 0], 0.2461, places=3)

        self.assertAlmostEqual(smooth_categories[0][0], 0.4436, places=3)
        self.assertAlmostEqual(smooth_categories[0][1], 0.5564, places=3)
        self.assertAlmostEqual(smooth_categories[1][0], 0.5564, places=3)
        self.assertAlmostEqual(smooth_categories[1][1], 0.4436, places=3)

    def test_batch_mixup_bypass(self):
        """Test batch mixup."""
        set_random_seed(0, deterministic=True)

        mixup = Mixup(probability=0.0, alpha=1.0, num_classes=2)
        images = [np.ones((32, 32, 3)), np.zeros((32, 32, 3))]
        categories = [np.array(1), np.array(0)]
        images, smooth_categories = mixup(images, categories)

        self.assertEqual(len(images), 2)
        self.assertEqual(len(smooth_categories), 2)

        self.assertAlmostEqual(smooth_categories[0][0], 0, places=3)
        self.assertAlmostEqual(smooth_categories[0][1], 1, places=3)
        self.assertAlmostEqual(smooth_categories[1][0], 1, places=3)
        self.assertAlmostEqual(smooth_categories[1][1], 0, places=3)
        