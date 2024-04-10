"""Testcases for DLA-UP."""

import unittest

from tests.util import generate_features
from vis4d.op.fpp.dla_up import DLAUp


class TestDLAUp(unittest.TestCase):
    """Testcases for DLA-UP."""

    def test_dlaup(self) -> None:
        """Testcase."""
        h, w = 128, 128
        inputs = generate_features(3, h, w, 6, 2, double_channels=True)
        dlaup = DLAUp(
            use_deformable_convs=True,
            start_level=2,
            in_channels=[3, 6, 12, 24, 48, 96],
        )
        outs = dlaup(inputs)
        self.assertEqual(len(outs), 1)
        self.assertEqual(outs[0].shape, (2, 12, h // 4, w // 4))
