"""Testcases for ViT op."""
from __future__ import annotations

import unittest

from tests.util import generate_features
from vis4d.op.base.vit import ViT


class TestViT(unittest.TestCase):
    """Testcases for ViT backbone."""

    def test_vit(self) -> None:
        """Testcase for VGG."""
        for vit_name, image_size in (
            ("vit_b_16", 512),
            ("vit_b_32", 512),
            ("vit_l_16", 512),
            ("vit_l_32", 512),
            ("vit_h_14", 224),
        ):
            self._test_vit(vit_name, image_size)

    def _test_vit(self, vit_name: str, image_size: int) -> None:
        """Testcase for ViT."""
        vit = ViT(vit_name, image_size=image_size, pretrained=False)

        test_images = generate_features(3, image_size, image_size, 1, 2)[0]
        out = vit(test_images)
        n_patches = 1 + (image_size // vit.patch_size) ** 2

        for i in range(2):
            self.assertEqual(out[i].shape[-1], vit.out_channels[i])
            self.assertEqual(out[i].shape[-2], n_patches)
