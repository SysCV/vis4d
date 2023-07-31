# pylint: disable=no-member
"""Tests for Mixup."""

import unittest

import numpy as np

from vis4d.common.util import set_random_seed
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.mixup import (
    GenMixupParameters,
    MixupBoxes2D,
    MixupCategories,
    MixupImages,
)
from vis4d.data.typing import DictData


class TestMixup(unittest.TestCase):
    """Tests for Mixup."""

    def test_mixup_no_padding(self):
        """Test batch mixup."""
        set_random_seed(0, deterministic=True)

        batch: list[DictData] = [
            {
                "images": np.ones((1, 32, 32, 3)).astype(np.float32),
                "categories": np.array([0.0, 1.0]),
                "boxes2d": np.array([[0.0, 0.0, 5.0, 5.0]], dtype=np.float32),
                "boxes2d_classes": np.array([0], dtype=np.int32),
            },
            {
                "images": np.zeros((1, 32, 32, 3)).astype(np.float32),
                "categories": np.array([1.0, 0.0]),
                "boxes2d": np.array(
                    [[3.0, 3.0, 10.0, 15.0]], dtype=np.float32
                ),
                "boxes2d_classes": np.array([1], dtype=np.int32),
            },
        ]

        tr1 = GenMixupParameters((32, 32), alpha=1.0)
        tr2 = MixupImages()
        tr3 = MixupCategories(num_classes=2, label_smoothing=0.1)
        tr4 = MixupBoxes2D()

        batch = tr1.apply_to_data(batch)
        batch = tr2.apply_to_data(batch)
        batch = tr3.apply_to_data(batch)
        batch = tr4.apply_to_data(batch)

        images = [data["images"] for data in batch]
        categories = [data["categories"] for data in batch]

        self.assertEqual(len(images), 2)
        self.assertEqual(len(categories), 2)
        self.assertEqual(categories[0].shape, (2,))
        self.assertEqual(categories[1].shape, (2,))
        self.assertEqual(images[0].shape, (1, 32, 32, 3))
        self.assertEqual(images[1].shape, (1, 32, 32, 3))

        for i in range(2):
            self.assertAlmostEqual(images[i][0, 0, 0, 0], 0.4491, places=3)
            self.assertAlmostEqual(categories[i][0], 0.5233, places=3)
            self.assertAlmostEqual(categories[i][1], 0.4267, places=3)

        assert (
            batch[0]["boxes2d"]
            == np.array(
                [[0.0, 0.0, 5.0, 5.0], [3.0, 3.0, 10.0, 15.0]],
                dtype=np.float32,
            )
        ).all()
        assert (
            batch[0]["boxes2d_classes"] == np.array([0, 1], dtype=np.int32)
        ).all()
        assert "boxes2d_track_ids" not in batch[0]

    def test_mixup_padding(self):
        """Test batch mixup."""
        set_random_seed(1, deterministic=True)

        batch: list[DictData] = [
            {
                "images": np.ones((1, 32, 32, 3)).astype(np.float32),
                "boxes2d": np.array(
                    [[5.0, 5.0, 10.0, 10.0]], dtype=np.float32
                ),
                "boxes2d_classes": np.array([0], dtype=np.int32),
                "boxes2d_track_ids": np.array([10], dtype=np.int32),
            },
            {
                "images": np.zeros((1, 32, 32, 3)).astype(np.float32),
                "boxes2d": np.array(
                    [[23.0, 13.0, 24.0, 15.0]], dtype=np.float32
                ),
                "boxes2d_classes": np.array([1], dtype=np.int32),
                "boxes2d_track_ids": np.array([20], dtype=np.int32),
            },
        ]

        tr1 = GenMixupParameters((64, 64), mixup_ratio_dist="const")
        tr2 = MixupImages()
        tr3 = MixupBoxes2D()

        batch = tr1.apply_to_data(batch)
        batch = tr2.apply_to_data(batch)
        batch = tr3.apply_to_data(batch)

        boxes = [data["boxes2d"] for data in batch]
        classes = [data["boxes2d_classes"] for data in batch]
        track_ids = [data["boxes2d_track_ids"] for data in batch]

        for box, cls, tid in zip(boxes, classes, track_ids):
            assert (
                box[0] == np.array([[5.0, 5.0, 10.0, 10.0]], dtype=np.float32)
            ).all()
            assert (
                box[1]
                == np.array([[30.0, 22.0, 32.0, 26.0]], dtype=np.float32)
            ).all()
            assert (cls[0] == np.array([0], dtype=np.int32)).all()
            assert (cls[1] == np.array([1], dtype=np.int32)).all()
            assert (tid[0] == np.array([10], dtype=np.int32)).all()
            assert (tid[1] == np.array([1020], dtype=np.int32)).all()

    def test_mixup_compose(self):
        """Test batch mixup using compose function."""
        set_random_seed(0, deterministic=True)

        batch: list[DictData] = [
            {
                "images": np.ones((1, 32, 32, 3)).astype(np.float32),
                "categories": np.array([0.0, 1.0]),
            },
            {
                "images": np.zeros((1, 32, 32, 3)).astype(np.float32),
                "categories": np.array([1.0, 0.0]),
            },
        ]

        tr1 = GenMixupParameters((32, 32), alpha=1.0)
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
        self.assertEqual(images[0].shape, (1, 32, 32, 3))
        self.assertEqual(images[1].shape, (1, 32, 32, 3))

        for i in range(2):
            self.assertAlmostEqual(images[i][0, 0, 0, 0], 0.4491, places=3)
            self.assertAlmostEqual(categories[i][0], 0.5233, places=3)
            self.assertAlmostEqual(categories[i][1], 0.4267, places=3)
