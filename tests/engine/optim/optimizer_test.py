"""Test cases for optimizer."""

from __future__ import annotations

import unittest

import torch
from ml_collections import ConfigDict
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.optim.optimizer import Optimizer

from tests.util import MockModel
from vis4d.config import class_config
from vis4d.engine.optim import LRSchedulerWrapper, PolyLR, set_up_optimizers
from vis4d.zoo.base import get_lr_scheduler_cfg, get_optimizer_cfg
from vis4d.zoo.typing import LrSchedulerConfig, ParamGroupCfg


def get_optimizer(
    model: nn.Module,
    optimizer: ConfigDict,
    lr_schedulers: list[LrSchedulerConfig] | None = None,
    param_groups: list[ParamGroupCfg] | None = None,
) -> tuple[list[Optimizer], list[LRSchedulerWrapper]]:
    """Get an optimizer for testing."""
    if lr_schedulers is None:
        lr_schedulers = [
            get_lr_scheduler_cfg(
                class_config(LinearLR, start_factor=0.1, total_iters=10),
                begin=0,
                end=10,
            ),
            get_lr_scheduler_cfg(
                class_config(PolyLR, max_steps=10, power=1.0), begin=10
            ),
        ]

    optimizer_cfg = get_optimizer_cfg(
        optimizer=optimizer,
        lr_schedulers=lr_schedulers,
        param_groups=param_groups,
    )
    return set_up_optimizers([optimizer_cfg], [model])


class TestOptimizer(unittest.TestCase):
    """Test cases for Optimizer."""

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
        optimizers, lr_scheulders = get_optimizer(
            MockModel(0), class_config(torch.optim.SGD, lr=0.01)
        )

        optimizer = optimizers[0]
        lr_scheulder = lr_scheulders[0]

        step = 0
        for epoch in range(20):
            if epoch in self.learning_rates:
                for _ in range(2):
                    self.assertAlmostEqual(
                        optimizer.param_groups[0]["lr"],
                        self.learning_rates[epoch],
                        places=5,
                    )
                    optimizer.step()
                    step += 1
                    lr_scheulder.step_on_batch(step)
            lr_scheulder.step(epoch)

    def test_optimizer_epoch_based_no_warmup(self) -> None:
        """Test the optimizer with epoch-based LR scheduling."""
        optimizers, lr_scheulders = get_optimizer(
            MockModel(0),
            class_config(torch.optim.SGD, lr=0.01),
            lr_schedulers=[
                get_lr_scheduler_cfg(
                    class_config(PolyLR, max_steps=20, power=1.0),
                    begin=0,
                    end=20,
                )
            ],
        )

        optimizer = optimizers[0]
        lr_scheulder = lr_scheulders[0]

        step = 0
        for epoch in range(20):
            for _ in range(2):
                if epoch in self.learning_rates_no_warmup:
                    self.assertAlmostEqual(
                        optimizer.param_groups[0]["lr"],
                        self.learning_rates_no_warmup[epoch],
                        places=5,
                    )
                optimizer.step()
                step += 1
                lr_scheulder.step_on_batch(step)
            lr_scheulder.step(epoch)

    def test_optimizer_batch_based(self) -> None:
        """Test the optimizer with batch-based LR scheduling."""
        optimizers, lr_scheulders = get_optimizer(
            MockModel(0),
            class_config(torch.optim.SGD, lr=0.01),
            lr_schedulers=[
                get_lr_scheduler_cfg(
                    class_config(LinearLR, start_factor=0.1, total_iters=10),
                    begin=0,
                    end=10,
                    epoch_based=False,
                ),
                get_lr_scheduler_cfg(
                    class_config(PolyLR, max_steps=10, power=1.0),
                    begin=10,
                    epoch_based=False,
                ),
            ],
        )

        optimizer = optimizers[0]
        lr_scheulder = lr_scheulders[0]

        step = 0
        for epoch in range(10):
            for _ in range(2):
                if step in self.learning_rates:
                    self.assertAlmostEqual(
                        optimizer.param_groups[0]["lr"],
                        self.learning_rates[step],
                        places=5,
                    )
                optimizer.step()
                step += 1
                lr_scheulder.step_on_batch(step)
            lr_scheulder.step(epoch)

    def test_optimizer_with_param_groups_cfg(self):
        """Test the optimizer with param_groups_cfg."""
        optimizers, lr_scheulders = get_optimizer(
            MockModel(0),
            class_config(torch.optim.AdamW, lr=0.01),
            param_groups=[
                ParamGroupCfg(custom_keys=["linear.weight"], lr_mult=0.1)
            ],
        )

        optimizer = optimizers[0]
        lr_scheulder = lr_scheulders[0]

        step = 0
        for epoch in range(20):
            if epoch in self.learning_rates:
                for _ in range(2):
                    self.assertAlmostEqual(
                        optimizer.param_groups[0]["lr"],
                        self.learning_rates[epoch] * 0.1,
                        places=5,
                    )
                    self.assertAlmostEqual(
                        optimizer.param_groups[1]["lr"],
                        self.learning_rates[epoch],
                        places=5,
                    )
                    optimizer.step()
                    step += 1
                    lr_scheulder.step_on_batch(step)
            lr_scheulder.step(epoch)
