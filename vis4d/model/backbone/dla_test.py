"""Testcases for DLA backbone."""
import unittest

from vis4d.unittest.utils import generate_input_sample

from .dla import DLA
from .neck import DLAUp


class TestDLA(unittest.TestCase):
    """Testcases for DLA backbone."""

    inputs = generate_input_sample(32, 32, 2, 2)

    def test_dla46_c(self) -> None:
        """Testcase for DLA46-C."""
        dla46_c = DLA(
            name="dla46_c",
            pixel_mean=(0.0, 0.0, 0.0),
            pixel_std=(1.0, 1.0, 1.0),
        )
        out = dla46_c(self.inputs)
        self.assertEqual(len(out), 6)
        channels = [16, 32, 64, 64, 128, 256]
        for i in range(6):
            feat = out[f"out{i}"]
            self.assertEqual(feat.shape[0], 2)
            self.assertEqual(feat.shape[1], channels[i])
            self.assertEqual(feat.shape[2], 32 / (2**i))
            self.assertEqual(feat.shape[3], 32 / (2**i))

    def test_dla46x_c(self) -> None:
        """Testcase for DLA46-X-C."""
        dla46x_c = DLA(
            name="dla46x_c",
            pixel_mean=(0.0, 0.0, 0.0),
            pixel_std=(1.0, 1.0, 1.0),
        )
        out = dla46x_c(self.inputs)
        self.assertEqual(len(out), 6)
        channels = [16, 32, 64, 64, 128, 256]
        for i in range(6):
            feat = out[f"out{i}"]
            self.assertEqual(feat.shape[0], 2)
            self.assertEqual(feat.shape[1], channels[i])
            self.assertEqual(feat.shape[2], 32 / (2**i))
            self.assertEqual(feat.shape[3], 32 / (2**i))

    def test_dla_custom(self) -> None:
        """Testcase for custom DLA + DLAUp Neck."""
        dla_custom = DLA(
            pixel_mean=(0.0, 0.0, 0.0),
            pixel_std=(1.0, 1.0, 1.0),
            levels=(1, 1, 1, 2, 2, 1),
            channels=(16, 32, 64, 128, 256, 512),
            block="BasicBlock",
            residual_root=True,
            neck=DLAUp(
                use_deformable_convs=False,
                start_level=2,
                in_channels=[16, 32, 64, 128, 256, 512],
            ),
        )
        out = dla_custom(self.inputs)
        self.assertEqual(tuple(out["out0"].shape[2:]), (8, 8))
        dla_custom.neck = None
        out = dla_custom(self.inputs)
        for i in range(6):
            feat = out[f"out{i}"]
            self.assertEqual(feat.shape[0], 2)
            self.assertEqual(feat.shape[1], 16 * (2**i))
            self.assertEqual(feat.shape[2], 32 / (2**i))
            self.assertEqual(feat.shape[3], 32 / (2**i))
