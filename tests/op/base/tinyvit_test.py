"""Testcases for TinyViT op."""
from __future__ import annotations

import unittest

from tests.util import generate_features
from vis4d.op.base import TinyViT


class TestViT(unittest.TestCase):
    """Testcases for ViT backbone."""

    def test_vit(self) -> None:
        """Testcase for VGG."""
        for vit_name, image_size in (
            ("tiny_vit_5m_224", 224),
            ("tiny_vit_11m_224", 224),
            ("tiny_vit_21m_224", 224),
        ):
            self._test_vit(vit_name, image_size)

    def _test_vit(self, vit_name: str, image_size: int) -> None:
        """Testcase for ViT."""
        model = TinyViT(vit_name, pretrained=False)
        test_images = generate_features(3, image_size, image_size, 1, 2)[0]
        feats = model(test_images)

        for i, feat in enumerate(feats):
            self.assertEqual(feat.shape[-1], model.out_channels[i])
