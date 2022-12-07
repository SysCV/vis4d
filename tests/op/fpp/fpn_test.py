"""Testcases for FPN."""
import unittest

from vis4d.op.fpp.fpn import FPN, LastLevelP6P7
from vis4d.unittest.util import generate_features


class TestFPN(unittest.TestCase):
    """Testcases for FPN."""

    def test_fpn(self) -> None:
        """Testcase for default FPN."""
        h, w, num_feats = 128, 128, 6
        inputs = generate_features(3, h, w, num_feats, 2, double_channels=True)
        fpn = FPN([12, 24, 48, 96], 48)
        outs = fpn(inputs)
        self.assertEqual(len(outs), num_feats + 1)
        for i, feat in enumerate(outs):
            self.assertEqual(feat.shape[0], 2)
            self.assertEqual(feat.shape[1], 48 if i >= 2 else 3 * (2**i))
            self.assertEqual(feat.shape[2], h / (2**i))
            self.assertEqual(feat.shape[3], w / (2**i))

    def test_p6p7(self) -> None:
        """Testcase for LastLevelP6P7."""
        h, w, num_feats = 128, 128, 6
        inputs = generate_features(3, h, w, num_feats, 2, double_channels=True)
        fpn = FPN([12, 24, 48, 96], 48, LastLevelP6P7(96, 48))
        outs = fpn(inputs)
        self.assertEqual(len(outs), num_feats + 2)
        for i, feat in enumerate(outs):
            self.assertEqual(feat.shape[0], 2)
            self.assertEqual(feat.shape[1], 48 if i >= 2 else 3 * (2**i))
            self.assertEqual(feat.shape[2], h / (2**i))
            self.assertEqual(feat.shape[3], w / (2**i))
