# pylint: disable=no-member
"""Test cases for photometric transforms."""
import copy
import unittest

import numpy as np
import torch
from PIL import Image

from tests.util import get_test_file
from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms.photometric import (
    ColorJitter,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    RandomHSV,
    RandomHue,
    RandomSaturation,
)


class TestPhotometric(unittest.TestCase):
    """Test Photometric transforms."""

    test_image = np.asarray(Image.open(get_test_file("image.jpg")))[
        None, ...
    ].astype(np.float32)

    def test_random_gamma(self):
        """Testcase for RandomGamma."""
        # test RGB
        data = {K.images: copy.deepcopy(self.test_image)}
        transform = RandomGamma((0.5, 0.5))
        data = transform.apply_to_data([data])[0]
        self.assertEqual(data[K.images].shape, (1, 230, 352, 3))
        assert np.allclose(
            data[K.images][0],
            torch.load(get_test_file("random_gamma_gt.npy")),
            atol=1e-4,
        )

        # test BGR
        data = {K.images: copy.deepcopy(self.test_image)[..., [2, 1, 0]]}
        transform = RandomGamma((0.5, 0.5), image_channel_mode="BGR")
        data = transform.apply_to_data([data])[0]
        self.assertEqual(data[K.images].shape, (1, 230, 352, 3))
        assert np.allclose(
            data[K.images][0][..., [2, 1, 0]],
            torch.load(get_test_file("random_gamma_gt.npy")),
            atol=1e-4,
        )

    def test_random_brightness(self):
        """Testcase for RandomBrightness."""
        data = {K.images: copy.deepcopy(self.test_image)}
        transform = RandomBrightness((0.5, 0.5))
        data = transform.apply_to_data([data])[0]
        self.assertEqual(data[K.images].shape, (1, 230, 352, 3))
        assert np.allclose(
            data[K.images][0],
            torch.load(get_test_file("random_brightness_gt.npy")),
            atol=1e-4,
        )

    def test_random_contrast(self):
        """Testcase for RandomContrast."""
        data = {K.images: copy.deepcopy(self.test_image)}
        transform = RandomContrast((0.5, 0.5))
        data = transform.apply_to_data([data])[0]
        self.assertEqual(data[K.images].shape, (1, 230, 352, 3))
        assert np.allclose(
            data[K.images][0],
            torch.load(get_test_file("random_contrast_gt.npy")),
            atol=1e-4,
        )

    def test_random_saturation(self):
        """Testcase for RandomSaturation."""
        data = {K.images: copy.deepcopy(self.test_image)}
        transform = RandomSaturation((0.5, 0.5))
        data = transform.apply_to_data([data])[0]
        self.assertEqual(data[K.images].shape, (1, 230, 352, 3))
        assert np.allclose(
            data[K.images][0],
            torch.load(get_test_file("random_saturation_gt.npy")),
            atol=1e-4,
        )

    def test_random_hue(self):
        """Testcase for RandomHue."""
        # test RGB
        data = {K.images: copy.deepcopy(self.test_image)}
        transform = RandomHue((0.05, 0.05))
        data = transform.apply_to_data([data])[0]
        self.assertEqual(data[K.images].shape, (1, 230, 352, 3))
        assert np.allclose(
            data[K.images][0],
            torch.load(get_test_file("random_hue_gt.npy")),
            atol=1e-4,
        )

        # test BGR
        data = {K.images: copy.deepcopy(self.test_image)[..., [2, 1, 0]]}
        transform = RandomHue((0.05, 0.05), image_channel_mode="BGR")
        data = transform.apply_to_data([data])[0]
        self.assertEqual(data[K.images].shape, (1, 230, 352, 3))
        assert np.allclose(
            data[K.images][0][..., [2, 1, 0]],
            torch.load(get_test_file("random_hue_gt.npy")),
            atol=1e-4,
        )

    def test_color_jitter(self):
        """Testcase for ColorJitter."""
        data = {K.images: copy.deepcopy(self.test_image)}
        transfrom = ColorJitter()
        data = transfrom.apply_to_data([data])[0]
        self.assertEqual(data[K.images].shape, (1, 230, 352, 3))

    def test_random_hsv(self):
        """Testcase for RandomHSV."""
        # test RGB image
        data = {K.images: copy.deepcopy(self.test_image)}
        transfrom = RandomHSV(image_channel_mode="RGB")
        data = transfrom.apply_to_data([data])[0]
        self.assertEqual(data[K.images].shape, (1, 230, 352, 3))

        # test BGR image
        data = {K.images: copy.deepcopy(self.test_image)[..., [2, 1, 0]]}
        transfrom = RandomHSV()
        data = transfrom.apply_to_data([data])[0]
        self.assertEqual(data[K.images].shape, (1, 230, 352, 3))
