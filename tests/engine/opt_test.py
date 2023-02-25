"""Test cases for optimizer."""
import unittest

import torch

from tests.util import MockModel
from vis4d.config.util import (
    class_config,
    delay_instantiation,
    instantiate_classes,
)
from vis4d.engine.opt import Optimizer
from vis4d.optim import LinearLRWarmup, PolyLR


class TestOptimizer(unittest.TestCase):
    """Test cases for callback functions."""

    learning_rates = {
        0: 0.001,
        4: 0.0046,
        10: 0.01,
        14: 0.005,
    }

    def test_optimizer_epoch_based(self) -> None:
        """Test the optimizer with epoch-based LR scheduling."""
        opt = class_config(torch.optim.SGD, lr=0.01)
        scheduler = class_config(PolyLR, max_steps=10, power=1.0)
        warmup = class_config(
            LinearLRWarmup, warmup_steps=10, warmup_ratio=0.1
        )
        optimizer_cfg = class_config(
            Optimizer,
            optimizer_cb=delay_instantiation(instantiable=opt),
            lr_scheduler_cb=delay_instantiation(instantiable=scheduler),
            lr_warmup=warmup,
            epoch_based=True,
        )
        optimizer = instantiate_classes(optimizer_cfg)
        optimizer.setup(MockModel(0))

        step = 0
        for epoch in range(20):
            optimizer.step_on_epoch(epoch)
            for _ in range(2):
                optimizer.step_on_batch(step)
                step += 1

            if epoch in self.learning_rates:
                self.assertAlmostEqual(
                    optimizer.optimizer.param_groups[0]["lr"],
                    self.learning_rates[epoch],
                    places=5,
                )

    def test_optimizer_batch_based(self) -> None:
        """Test the optimizer with batch-based LR scheduling."""
        opt = class_config(torch.optim.SGD, lr=0.01)
        scheduler = class_config(PolyLR, max_steps=10, power=1.0)
        warmup = class_config(
            LinearLRWarmup, warmup_steps=10, warmup_ratio=0.1
        )
        optimizer_cfg = class_config(
            Optimizer,
            optimizer_cb=delay_instantiation(instantiable=opt),
            lr_scheduler_cb=delay_instantiation(instantiable=scheduler),
            lr_warmup=warmup,
            epoch_based=False,
        )
        optimizer = instantiate_classes(optimizer_cfg)
        optimizer.setup(MockModel(0))

        step = 0
        for epoch in range(10):
            optimizer.step_on_epoch(epoch)
            for _ in range(2):
                optimizer.step_on_batch(step)
                if step in self.learning_rates:
                    self.assertAlmostEqual(
                        optimizer.optimizer.param_groups[0]["lr"],
                        self.learning_rates[step],
                        places=5,
                    )
                step += 1
