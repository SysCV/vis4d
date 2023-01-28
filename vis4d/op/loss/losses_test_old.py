"""Testcases for losses."""
# TODO, move to test
import unittest

import torch

from .box3d_uncertainty_loss import Box3DUncertaintyLoss
from .embedding_distance import EmbeddingDistanceLoss
from .multi_pos_cross_entropy import MultiPosCrossEntropyLoss


class TestQDLosses(unittest.TestCase):
    """Testcases for loss functions in qdtrack."""

    pred = torch.Tensor(
        [
            [
                0.1747,
                0.0020,
                0.0909,
                0.2534,
                0.4916,
                -0.4987,
                0.0686,
                0.3338,
                0.3276,
                0.4907,
            ],
            [
                0.4417,
                -0.0452,
                0.0429,
                -0.4182,
                0.4432,
                0.3206,
                0.2496,
                0.0600,
                -0.2923,
                -0.0241,
            ],
            [
                0.3914,
                0.0224,
                0.4258,
                -0.3901,
                -0.1909,
                -0.3636,
                0.1434,
                0.4838,
                0.4997,
                -0.3403,
            ],
            [
                0.3544,
                0.0173,
                -0.0708,
                0.2005,
                0.2115,
                0.4953,
                0.0423,
                0.2846,
                0.1164,
                0.2775,
            ],
            [
                -0.2481,
                0.4610,
                0.2738,
                -0.4015,
                -0.1024,
                0.0089,
                -0.1319,
                0.1324,
                0.4087,
                -0.4041,
            ],
            [
                -0.4779,
                -0.1742,
                -0.0917,
                -0.0751,
                0.4192,
                0.2384,
                -0.1329,
                -0.1022,
                -0.2148,
                -0.0591,
            ],
            [
                0.3726,
                0.1764,
                -0.2545,
                -0.1851,
                0.1749,
                0.2817,
                -0.2632,
                0.2476,
                0.1992,
                -0.3983,
            ],
            [
                -0.3153,
                -0.3166,
                0.2126,
                0.2174,
                -0.1625,
                -0.2301,
                -0.3114,
                -0.2889,
                0.4236,
                -0.4473,
            ],
            [
                0.0147,
                0.0913,
                -0.1837,
                -0.1589,
                0.1108,
                -0.2437,
                0.3706,
                0.1597,
                0.0391,
                0.2575,
            ],
            [
                -0.4839,
                0.0437,
                -0.1093,
                0.1051,
                -0.1232,
                -0.1602,
                0.1795,
                -0.0981,
                0.4120,
                -0.0017,
            ],
        ]
    )

    target = torch.Tensor(
        [
            [1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
            [0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
            [0, 1, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 1, 0],
        ]
    )

    def test_embed_distance(self) -> None:
        """Testcase for embedding distance loss."""
        loss = EmbeddingDistanceLoss(
            neg_pos_ub=0.5,
            pos_margin=0.5,
            neg_margin=0.5,
            hard_mining=True,
        )
        x = loss(self.pred, self.target)
        self.assertTrue(abs(x - 0.6667) < 1e-4)

        loss = EmbeddingDistanceLoss(
            neg_pos_ub=0.5,
            pos_margin=0.5,
            neg_margin=0.5,
            hard_mining=False,
        )
        x = loss(self.pred, self.target)
        self.assertTrue(abs(x - 0.6667) < 1e-4)

    def test_multipos_crossentropy(self) -> None:
        """Testcase for multi positive cross-entropy loss."""
        loss = MultiPosCrossEntropyLoss()
        x = loss(self.pred, self.target, reduction_override="sum")
        self.assertTrue(abs(x - 34.0866) < 1e-4)
        x = loss(self.pred, self.target, reduction_override="mean")
        self.assertTrue(abs(x - 3.4087) < 1e-4)


class TestLossBox3D(unittest.TestCase):
    """Testclass for Box3d loss."""

    def test_box3d_loss(self) -> None:
        """Testcase for box3d loss."""
        box3d_loss = Box3DUncertaintyLoss()
        loss_dict = box3d_loss(
            torch.empty(0, 5), torch.empty(0, 5), torch.empty(0)
        )
        for v in loss_dict.values():
            self.assertEqual(v.item(), 0.0)
