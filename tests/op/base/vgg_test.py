"""Testcases for VGG base model."""

from __future__ import annotations

import unittest

from tests.util import generate_features
from vis4d.op.base.vgg import VGG


class TestVGG(unittest.TestCase):
    """Testcases for VGG base model."""

    def test_vgg(self) -> None:
        """Testcase for VGG."""
        for vgg_name in ("vgg11", "vgg13", "vgg16", "vgg19"):
            self._test_vgg(vgg_name)
            self._test_vgg(vgg_name + "_bn")

    def _test_vgg(self, vgg_name: str) -> None:
        """Testcase for VGG."""
        vgg = VGG(vgg_name, pretrained=False)

        test_images = generate_features(3, 64, 64, 1, 2)[0]
        out = vgg(test_images)

        channels = [3, 3, 64, 128, 256, 512, 512]
        self.assertEqual(vgg.out_channels, channels)
        self.assertEqual(len(out), 7)

        self.assertEqual(out[0].shape[0], 2)
        self.assertEqual(out[0].shape[1], 3)
        self.assertEqual(out[0].shape[2], 64)
        self.assertEqual(out[0].shape[3], 64)

        for i in range(1, 7):
            feat = out[i]
            self.assertEqual(feat.shape[0], 2)
            self.assertEqual(feat.shape[1], channels[i])
            self.assertEqual(feat.shape[2], 64 / (2 ** (i - 1)))
            self.assertEqual(feat.shape[3], 64 / (2 ** (i - 1)))
