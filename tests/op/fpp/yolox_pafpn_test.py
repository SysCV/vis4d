"""Testcases for YOLOXPAFPN."""

import unittest

import torch

from tests.util import fill_weights, generate_features_determ, get_test_file
from vis4d.op.fpp.yolox_pafpn import YOLOXPAFPN


class TestYOLOXPAFPN(unittest.TestCase):
    """Testcases for YOLOXPAFPN."""

    def test_yolox_pafpn(self):
        """Test YOLOXPAFPN."""
        size = 32
        features = generate_features_determ(64, size, size, 3, 2, True)
        yolox_pafpn = YOLOXPAFPN([64, 128, 256], 64, start_index=0).eval()
        fill_weights(yolox_pafpn, 1.0)
        out = yolox_pafpn(features)
        gt_out = torch.load(get_test_file("yolox_pafpn.pt"))
        self.assertTrue(len(out) == 3)
        for i, f in enumerate(out):
            self.assertEqual(f.shape[:2], (2, 64))
            self.assertEqual(f.shape[2:], (size // 2**i, size // 2**i))
            self.assertTrue(torch.isclose(f, gt_out[i], atol=1e-3).all())
