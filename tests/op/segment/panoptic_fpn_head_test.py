"""Testcases for panoptic fpn head."""
import unittest

from vis4d.op.segment.panoptic_fpn_head import PanopticFPNHead
from vis4d.unittest.util import generate_features


class TestPanopticFPNHead(unittest.TestCase):
    """Testcases for PanopticFPNHead."""

    def test_forward(self) -> None:
        """Testcase for forward pass."""
        batch_size, h, w, inc, num_cls = 2, 64, 128, 256, 53
        test_features = [None, None] + generate_features(
            inc, h, w, 5, batch_size
        )
        pan_head = PanopticFPNHead(num_cls)
        segms = pan_head(test_features)
        assert len(segms) == batch_size
        assert segms.shape[1:] == (53 + 1, h * 4, w * 4)

        batch_size, h, w, inc, num_cls = 2, 32, 64, 128, 20
        test_features = [None, None] + generate_features(
            inc, h, w, 4, batch_size
        )
        pan_head = PanopticFPNHead(20, 128, 64, 3)
        segms = pan_head(test_features)
        assert len(segms) == batch_size
        assert segms.shape[1:] == (num_cls + 1, h * 4, w * 4)
