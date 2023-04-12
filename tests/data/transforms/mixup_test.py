# tests/transforms/test_mixup.py
"""Tests for Mixup."""

import unittest

import numpy as np

from vis4d.common.util import set_random_seed
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.mixup import (
    GenerateMixupParameters,
    MixupCategories,
    MixupImages,
)
from vis4d.data.typing import DictData


class TestMixup(unittest.TestCase):
    """Tests for Mixup."""

    def test_batch_mixup(self):
        """Test batch mixup."""
        set_random_seed(0, deterministic=True)

        batch: DictData = [
            dict(
                images=np.ones((32, 32, 3)).astype(np.float32),
                categories=np.array([0.0, 1.0]),
            ),
            dict(
                images=np.zeros((32, 32, 3)).astype(np.float32),
                categories=np.array([1.0, 0.0]),
            ),
        ]

        tr1 = GenerateMixupParameters(alpha=1.0)
        tr2 = MixupImages()
        tr3 = MixupCategories(num_classes=2, label_smoothing=0.1)

        batch = tr1.apply_to_data(batch)  # pylint: disable=no-member
        batch = tr2.apply_to_data(batch)  # pylint: disable=no-member
        batch = tr3.apply_to_data(batch)  # pylint: disable=no-member

        images = [data["images"] for data in batch]
        categories = [data["categories"] for data in batch]

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

    def test_batch_mixup_compose(self):
        """Test batch mixup using compose function."""
        set_random_seed(0, deterministic=True)

        batch: DictData = [
            dict(
                images=np.ones((32, 32, 3)).astype(np.float32),
                categories=np.array([0.0, 1.0]),
            ),
            dict(
                images=np.zeros((32, 32, 3)).astype(np.float32),
                categories=np.array([1.0, 0.0]),
            ),
        ]

        tr1 = GenerateMixupParameters(alpha=1.0)
        tr2 = MixupImages()
        tr3 = MixupCategories(num_classes=2, label_smoothing=0.1)
        tr = compose([tr1, tr2, tr3])

        batch = tr(batch)
        images = [data["images"] for data in batch]
        categories = [data["categories"] for data in batch]

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
