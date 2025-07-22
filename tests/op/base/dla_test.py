"""Testcases for DLA base model."""

import unittest

from tests.util import generate_features
from vis4d.op.base.dla import DLA


class TestDLA(unittest.TestCase):
    """Testcases for DLA base model."""

    inputs = generate_features(3, 32, 32, 1, 2)[0]

    def test_dla46_c(self) -> None:
        """Testcase for DLA46-C."""
        dla46_c = DLA(name="dla46_c")
        out = dla46_c(self.inputs)
        self.assertEqual(len(out), 6)
        channels = [3, 3, 64, 64, 128, 256]
        for i in range(2, 6):
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
        channels = [3, 3, 64, 64, 128, 256]
        for i in range(2, 6):
            feat = out[i]
            self.assertEqual(feat.shape[0], 2)
            self.assertEqual(feat.shape[1], channels[i])
            self.assertEqual(feat.shape[2], 32 / (2**i))
            self.assertEqual(feat.shape[3], 32 / (2**i))
