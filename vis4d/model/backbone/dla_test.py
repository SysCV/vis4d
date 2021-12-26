"""Testcases for DLA backbone."""
import unittest

from vis4d.unittest.utils import generate_input_sample

from .dla import DLA, DLAConfig


class TestDLA(unittest.TestCase):
    """Testcases for DLA backbone."""

    inputs = generate_input_sample(32, 32, 2, 2)

    def test_dla46_c(self) -> None:
        """Testcase for DLA46-C."""
        cfg = DLAConfig(type="DLA", name="dla46_c")
        dla46_c = DLA(cfg)
        out = dla46_c(self.inputs)
        self.assertEqual(len(out), 6)
        channels = [16, 32, 64, 64, 128, 256]
        for i in range(6):
            feat = out[f"out{i}"]
            self.assertEqual(feat.shape[0], 2)
            self.assertEqual(feat.shape[1], channels[i])
            self.assertEqual(feat.shape[2], 32 / (2 ** i))
            self.assertEqual(feat.shape[3], 32 / (2 ** i))

    def test_dla46x_c(self) -> None:
        """Testcase for DLA46-X-C."""
        cfg = DLAConfig(type="DLA", name="dla46x_c")
        dla46x_c = DLA(cfg)
        out = dla46x_c(self.inputs)
        self.assertEqual(len(out), 6)
        channels = [16, 32, 64, 64, 128, 256]
        for i in range(6):
            feat = out[f"out{i}"]
            self.assertEqual(feat.shape[0], 2)
            self.assertEqual(feat.shape[1], channels[i])
            self.assertEqual(feat.shape[2], 32 / (2 ** i))
            self.assertEqual(feat.shape[3], 32 / (2 ** i))

    def test_dla_custom(self) -> None:
        """Testcase for custom DLA."""
        cfg = DLAConfig(
            type="DLA",
            levels=(1, 1, 1, 2, 2, 1),
            channels=(16, 32, 64, 128, 256, 512),
            block="BasicBlock",
            residual_root=True,
        )
        dla_custom = DLA(cfg)
        out = dla_custom(self.inputs)
        for i in range(6):
            feat = out[f"out{i}"]
            self.assertEqual(feat.shape[0], 2)
            self.assertEqual(feat.shape[1], 16 * (2 ** i))
            self.assertEqual(feat.shape[2], 32 / (2 ** i))
            self.assertEqual(feat.shape[3], 32 / (2 ** i))
