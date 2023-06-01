# pylint: disable=no-member
"""Test Pad transform."""
import unittest

import numpy as np

from vis4d.data.transforms.pad import PadImages, PadSegMasks


class TestPad(unittest.TestCase):
    """Unit tests for Pad transforms."""

    def test_pad_images(self) -> None:
        """Test PadImages."""
        images = [
            np.random.rand(1, 10, 10, 3).astype(np.float32),
            np.random.rand(1, 15, 15, 3).astype(np.float32),
        ]
        pad_images = PadImages(stride=4)
        images, hw = pad_images(images)
        self.assertEqual(images[0].shape, (1, 16, 16, 3))
        self.assertEqual(images[1].shape, (1, 16, 16, 3))
        self.assertEqual(hw, [(16, 16), (16, 16)])
        self.assertTrue((images[0][0, 10:, 10:] == 0).all())
        self.assertTrue((images[1][0, 15:, 15:] == 0).all())

        images = [
            np.random.rand(1, 10, 10, 3).astype(np.float32),
            np.random.rand(1, 15, 15, 3).astype(np.float32),
        ]
        pad_images = PadImages(shape=(17, 17))
        images, hw = pad_images(images)
        self.assertEqual(images[0].shape, (1, 17, 17, 3))
        self.assertEqual(images[1].shape, (1, 17, 17, 3))
        self.assertEqual(hw, [(17, 17), (17, 17)])
        self.assertTrue((images[0][0, 10:, 10:] == 0).all())
        self.assertTrue((images[1][0, 15:, 15:] == 0).all())

    def test_pad_seg_masks(self) -> None:
        """Test PadSegMasks."""
        masks = [
            np.random.rand(10, 10).astype(np.uint8),
            np.random.rand(15, 15).astype(np.uint8),
        ]
        pad_seg_masks = PadSegMasks(stride=4)
        masks = pad_seg_masks(masks)
        self.assertEqual(masks[0].shape, (16, 16))
        self.assertEqual(masks[1].shape, (16, 16))
        self.assertTrue((masks[0][10:, 10:] == 255).all())
        self.assertTrue((masks[1][15:, 15:] == 255).all())

        masks = [
            np.random.rand(10, 10).astype(np.uint8),
            np.random.rand(15, 15).astype(np.uint8),
        ]
        pad_seg_masks = PadSegMasks(shape=(17, 17))
        masks = pad_seg_masks(masks)
        self.assertEqual(masks[0].shape, (17, 17))
        self.assertEqual(masks[1].shape, (17, 17))
        self.assertTrue((masks[0][10:, 10:] == 255).all())
        self.assertTrue((masks[1][15:, 15:] == 255).all())
