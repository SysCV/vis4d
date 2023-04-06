"""Testcases for Semantic FPN Head."""
import unittest

from tests.util import generate_features
from vis4d.op.segment.semantic_fpn import SemanticFPNHead


class TestSemanticFPNHead(unittest.TestCase):
    """Testcases for SemanticFPNHead."""

    def test_forward(self) -> None:
        """Testcase for forward pass."""
        batch_size, h, w, inc, num_cls = 2, 64, 128, 256, 53
        test_features = [None, None] + generate_features(
            inc, h, w, 5, batch_size
        )
        pan_head = SemanticFPNHead(num_cls)
        segms = pan_head(test_features).outputs
        assert len(segms) == batch_size
        assert segms.shape[1:] == (num_cls, h, w)

        batch_size, h, w, inc, num_cls = 2, 32, 64, 128, 20
        test_features = [None, None] + generate_features(
            inc, h, w, 4, batch_size
        )
        pan_head = SemanticFPNHead(20, 128, 64, 3)
        segms = pan_head(test_features).outputs
        assert len(segms) == batch_size
        assert segms.shape[1:] == (num_cls, h // 2, w // 2)
