"""Testcases for quasi-dense losses."""
import unittest

import torch

from vis4d.op.loss.box3d_uncertainty_loss import Box3DUncertaintyLoss


class TestQD3DTBox3DLoss(unittest.TestCase):
    """Testclass for Box3d loss of QD-3DT."""

    def test_box3d_loss(self) -> None:
        """Testcase for box3d loss."""
        box3d_loss = Box3DUncertaintyLoss()
        loss_dict = box3d_loss(
            torch.empty(0, 5), torch.empty(0, 5), torch.empty(0)
        )
        for v in loss_dict.values():
            self.assertEqual(v.item(), 0.0)
