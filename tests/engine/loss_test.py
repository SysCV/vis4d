"""Test for engine.loss."""

import unittest

import torch

from vis4d.engine.loss import WeightedMultiLoss


class WeightedLossTest(unittest.TestCase):
    """Weighted loss test class."""

    def test_forward(self) -> None:
        """Test forward."""
        loss = WeightedMultiLoss(
            [
                {"loss": torch.nn.MSELoss(), "weight": 0.7},
                {"loss": torch.nn.L1Loss(), "weight": 0.3},
            ]
        )
        x = torch.rand(2, 3, 4, 5)
        y = torch.rand(2, 3, 4, 5)
        losses = loss(input=x, target=y)
        total_loss = sum(losses.values())

        self.assertAlmostEqual(
            total_loss.item(),
            0.7 * torch.nn.MSELoss()(x, y).item()
            + 0.3 * torch.nn.L1Loss()(x, y).item(),
            places=3,
        )
