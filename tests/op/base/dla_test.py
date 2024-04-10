"""Testcases for DLA base model."""

import unittest

from tests.util import generate_features
from vis4d.op.base.dla import DLA


class TestDLA(unittest.TestCase):
    """Testcases for DLA base model."""

    inputs = generate_features(3, 32, 32, 1, 2)[0]

    def test_dla46_c(self) -> None:
        """Testcase for DLA46-C."""
        dla46_c = DLA(
            name="dla46_c",
            # weights=
            # "http://dl.yf.io/dla/models/imagenet/dla46_c-2bfd52c3.pth",
        )
        out = dla46_c(self.inputs)
        self.assertEqual(len(out), 6)
        channels = [16, 32, 64, 64, 128, 256]
        for i in range(6):
            feat = out[i]
            self.assertEqual(feat.shape[0], 2)
            self.assertEqual(feat.shape[1], channels[i])
            self.assertEqual(feat.shape[2], 32 / (2**i))
            self.assertEqual(feat.shape[3], 32 / (2**i))

    def test_dla46x_c(self) -> None:
        """Testcase for DLA46-X-C."""
        dla46x_c = DLA(name="dla46x_c")
        out = dla46x_c(self.inputs)
        self.assertEqual(len(out), 6)
        channels = [16, 32, 64, 64, 128, 256]
        for i in range(6):
            feat = out[i]
            self.assertEqual(feat.shape[0], 2)
            self.assertEqual(feat.shape[1], channels[i])
            self.assertEqual(feat.shape[2], 32 / (2**i))
            self.assertEqual(feat.shape[3], 32 / (2**i))

    def test_dla_custom(self) -> None:
        """Testcase for custom DLA."""
        dla_custom = DLA(
            levels=(1, 1, 1, 2, 2, 1),
            channels=(16, 32, 64, 128, 256, 512),
            block="BasicBlock",
            residual_root=True,
        )
        out = dla_custom(self.inputs)
        self.assertEqual(tuple(out[2].shape[2:]), (8, 8))
        dla_custom = DLA(
            levels=(1, 1, 1, 2, 2, 1),
            channels=(16, 32, 64, 128, 256, 512),
            block="BasicBlock",
            residual_root=False,
        )
        out = dla_custom(self.inputs)
        self.assertEqual(tuple(out[2].shape[2:]), (8, 8))
        out = dla_custom(self.inputs)
        for i in range(6):
            feat = out[i]
            self.assertEqual(feat.shape[0], 2)
            self.assertEqual(feat.shape[1], 16 * (2**i))
            self.assertEqual(feat.shape[2], 32 / (2**i))
            self.assertEqual(feat.shape[3], 32 / (2**i))
