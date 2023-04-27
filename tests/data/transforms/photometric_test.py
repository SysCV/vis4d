# pylint: disable=no-member
"""Test cases for photometric transforms."""
import copy
import unittest

import numpy as np
import torch
from PIL import Image

from tests.util import get_test_file
from vis4d.data.transforms.photometric import (
    ColorJitter,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    RandomHue,
    RandomSaturation,
)


class TestPhotometric(unittest.TestCase):
    """Test Photometric transforms."""

    test_image = np.asarray(Image.open(get_test_file("image.jpg")))[None, ...]

    def test_random_gamma(self):
        """Testcase for RandomGamma."""
        data = {"images": copy.deepcopy(self.test_image)}
        tr1 = RandomGamma((0.5, 0.5))
        data = tr1.apply_to_data(data)
        self.assertEqual(data["images"].shape, (1, 230, 352, 3))
        assert np.allclose(
            data["images"][0],
            torch.load(get_test_file("random_gamma_gt.npy")),
            atol=1e-4,
        )

    def test_random_brightness(self):
        """Testcase for RandomBrightness."""
        data = {"images": copy.deepcopy(self.test_image)}
        tr1 = RandomBrightness((0.5, 0.5))
        data = tr1.apply_to_data(data)
        self.assertEqual(data["images"].shape, (1, 230, 352, 3))
        assert np.allclose(
            data["images"][0],
            torch.load(get_test_file("random_brightness_gt.npy")),
            atol=1e-4,
        )

    def test_random_contrast(self):
        """Testcase for RandomContrast."""
        data = {"images": copy.deepcopy(self.test_image)}
        tr1 = RandomContrast((0.5, 0.5))
        data = tr1.apply_to_data(data)
        self.assertEqual(data["images"].shape, (1, 230, 352, 3))
        assert np.allclose(
            data["images"][0],
            torch.load(get_test_file("random_contrast_gt.npy")),
            atol=1e-4,
        )

    def test_random_saturation(self):
        """Testcase for RandomSaturation."""
        data = {"images": copy.deepcopy(self.test_image)}
        tr1 = RandomSaturation((0.5, 0.5))
        data = tr1.apply_to_data(data)
        self.assertEqual(data["images"].shape, (1, 230, 352, 3))
        assert np.allclose(
            data["images"][0],
            torch.load(get_test_file("random_saturation_gt.npy")),
            atol=1e-4,
        )

    def test_random_hue(self):
        """Testcase for RandomHue."""
        data = {"images": copy.deepcopy(self.test_image)}
        tr1 = RandomHue((0.05, 0.05))
        data = tr1.apply_to_data(data)
        self.assertEqual(data["images"].shape, (1, 230, 352, 3))
        assert np.allclose(
            data["images"][0],
            torch.load(get_test_file("random_hue_gt.npy")),
            atol=1e-4,
        )

    def test_color_jitter(self):
        """Testcase for ColorJitter."""
        data = {"images": copy.deepcopy(self.test_image)}
        tr1 = ColorJitter()
        data = tr1.apply_to_data(data)
        self.assertEqual(data["images"].shape, (1, 230, 352, 3))
        # Image.fromarray(data['images'][0]).save("test.jpg")
