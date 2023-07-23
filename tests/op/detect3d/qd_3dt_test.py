"""Testcases for quasi-dense losses."""
import unittest

import torch

from vis4d.op.detect3d.qd_3dt import Box3DUncertaintyLoss


class TestQD3DTBox3DLoss(unittest.TestCase):
    """Testclass for Box3d loss of QD-3DT."""

    def test_box3d_loss(self) -> None:
        """Testcase for box3d loss."""
        box3d_loss = Box3DUncertaintyLoss()

        loss_dict = box3d_loss(
            torch.empty(0, 10, 12), torch.empty(0, 12), torch.empty(0)
        )
        for v in loss_dict.values():
            self.assertEqual(v.item(), 0.0)

        loss_dict = box3d_loss(
            torch.ones((4, 10, 12)),
            torch.ones((4, 12)),
            torch.ones((4), dtype=torch.int64),
        )
        for k, v in loss_dict.items():
            if k != "loss_rot3d":
                self.assertEqual(v.item(), 0.0)
            else:
                self.assertAlmostEqual(v.item(), 0.863, delta=1e-3)
