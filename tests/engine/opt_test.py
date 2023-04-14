"""Test cases for optimizer."""
from __future__ import annotations

import unittest

import torch
from torch import nn

from tests.util import MockModel
from vis4d.config.default.optimizer import get_optimizer_config
from vis4d.config.util import ConfigDict, class_config
from vis4d.engine.opt import Optimizer, set_up_optimizers
from vis4d.optim import LinearLRWarmup, PolyLR


def get_optimizer(
    model: nn.Module = MockModel(0),
    optimizer: ConfigDict = class_config(torch.optim.SGD, lr=0.01),
    lr_scheduler: None
    | ConfigDict = class_config(PolyLR, max_steps=10, power=1.0),
    lr_warmup: None
    | ConfigDict = class_config(
        LinearLRWarmup, warmup_steps=10, warmup_ratio=0.1
    ),
    epoch_based_lr: bool = True,
    epoch_based_warmup: bool = True,
) -> Optimizer:
    """Get an optimizer for testing."""
    optimizer_cfg = get_optimizer_config(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_warmup=lr_warmup,
        epoch_based_lr=epoch_based_lr,
        epoch_based_warmup=epoch_based_warmup,
    )
    return set_up_optimizers([optimizer_cfg], model)[0]


class TestOptimizer(unittest.TestCase):
    """Test cases for callback functions."""

    learning_rates = {
        0: 0.001,
        4: 0.0046,
        10: 0.01,
        11: 0.009,
        12: 0.008,
        18: 0.002,
        19: 0.001,
    }

    learning_rates_no_warmup = {
        0: 0.01,
        1: 0.0095,
        2: 0.009,
        18: 0.001,
        19: 0.0005,
    }

    def test_optimizer_epoch_based(self) -> None:
        """Test the optimizer with epoch-based LR scheduling."""
        optimizer = get_optimizer()

        step = 0
        for epoch in range(20):
            for _ in range(2):
                if epoch in self.learning_rates:
                    self.assertAlmostEqual(
                        optimizer.optimizer.param_groups[0]["lr"],
                        self.learning_rates[epoch],
                        places=5,
                    )
                optimizer.step_on_batch(step)
                step += 1

            optimizer.step_on_epoch(epoch)

    def test_optimizer_epoch_based_no_warmup(self) -> None:
        """Test the optimizer with epoch-based LR scheduling."""
        optimizer = get_optimizer(
            lr_scheduler=class_config(PolyLR, max_steps=20, power=1.0),
            lr_warmup=None,
        )

        step = 0
        for epoch in range(20):
            for _ in range(2):
                if epoch in self.learning_rates_no_warmup:
                    self.assertAlmostEqual(
                        optimizer.optimizer.param_groups[0]["lr"],
                        self.learning_rates_no_warmup[epoch],
                        places=5,
                    )
                optimizer.step_on_batch(step)
                step += 1

            optimizer.step_on_epoch(epoch)

    def test_optimizer_batch_based(self) -> None:
        """Test the optimizer with batch-based LR scheduling."""
        optimizer = get_optimizer(
            epoch_based_lr=False, epoch_based_warmup=False
        )

        step = 0
        for epoch in range(10):
            for _ in range(2):
                if step in self.learning_rates:
                    self.assertAlmostEqual(
                        optimizer.optimizer.param_groups[0]["lr"],
                        self.learning_rates[step],
                        places=5,
                    )
                optimizer.step_on_batch(step)
                step += 1

            optimizer.step_on_epoch(epoch)
