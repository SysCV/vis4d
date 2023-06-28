"""Test cases for learning rate schedulers."""
import copy
import math
import unittest

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import LRScheduler
from torch.testing import assert_close

from vis4d.engine.optim.scheduler import ConstantLR, PolyLR, QuadraticLRWarmup


class ToyModel(torch.nn.Module):
    """Toy model for testing."""

    def __init__(self):
        """Init."""
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        """Forward."""
        return self.conv2(F.relu(self.conv1(x)))


def _test_scheduler_value(
    optimizer,
    schedulers,
    targets,
    epochs=10,
    param_name="lr",
    step_kwargs=None,
):
    """Test the value of the scheduler."""
    if isinstance(schedulers, LRScheduler):
        schedulers = [schedulers]
    if step_kwargs is None:
        step_kwarg = [{} for _ in range(len(schedulers))]
        step_kwargs = [step_kwarg for _ in range(epochs)]
    else:  # step_kwargs is not None
        assert len(step_kwargs) == epochs
        assert len(step_kwargs[0]) == len(schedulers)
    for epoch in range(epochs):
        for param_group, target in zip(optimizer.param_groups, targets):
            assert_close(
                target[epoch],
                param_group[param_name],
                msg="{} is wrong in epoch {}: expected {}, got {}".format(  # pylint: disable=consider-using-f-string,line-too-long
                    param_name, epoch, target[epoch], param_group[param_name]
                ),
                atol=1e-5,
                rtol=0,
            )
        _ = [
            scheduler.step(**step_kwargs[epoch][i])
            for i, scheduler in enumerate(schedulers)
        ]


class TestScheduler(unittest.TestCase):
    """Test cases for Scheduler."""

    def setUp(self) -> None:
        """Set up."""
        model = ToyModel()
        self.lr = 0.05
        self.l2_mult = 10
        self.optimizer = optim.SGD(
            [
                {"params": model.conv1.parameters()},
                {
                    "params": model.conv2.parameters(),
                    "lr": self.lr * self.l2_mult,
                },
            ],
            lr=self.lr,
            momentum=0.01,
            weight_decay=5e-4,
        )

    def test_constant_lr_scheduler(self):
        """Test case for ConstantLR."""
        epochs, t = 12, 10
        single_targets = [
            self.lr * 1 / 3.0 if x < t else self.lr for x in range(epochs)
        ]
        targets = [
            single_targets,
            [x * self.l2_mult for i, x in enumerate(single_targets)],
        ]
        optimizer = copy.deepcopy(self.optimizer)
        scheduler = ConstantLR(optimizer, max_steps=t)
        _test_scheduler_value(optimizer, scheduler, targets, epochs)

    def test_poly_lr_scheduler(self):
        """Test case for PolyLR."""
        epochs, t = 12, 10
        min_lr, power = 0.0001, 0.9
        single_targets, l2_targets = [self.lr], [self.lr * self.l2_mult]
        for x in range(1, epochs):
            if x < t:
                single_targets.append(
                    (single_targets[-1] - min_lr)
                    * (1 - 1 / (t - x + 1)) ** power
                    + min_lr
                )
                l2_targets.append(
                    (l2_targets[-1] - min_lr) * (1 - 1 / (t - x + 1)) ** power
                    + min_lr
                )
            else:
                single_targets.append(min_lr)
                l2_targets.append(min_lr)
        targets = [single_targets, l2_targets]
        optimizer = copy.deepcopy(self.optimizer)
        scheduler = PolyLR(optimizer, max_steps=t, min_lr=min_lr, power=power)
        _test_scheduler_value(optimizer, scheduler, targets, epochs)

    def test_quadratic_lr_warmup_scheduler(self):
        """Test case for QuadraticLRWarmup."""
        epochs, t = 12, 10
        single_targets, l2_targets = [0.0], [0.0]
        for x in range(epochs):
            if x < t:
                single_targets.append(
                    single_targets[-1] + self.lr * (2 * x + 1) / t**2
                )
                l2_targets.append(
                    l2_targets[-1]
                    + self.lr * self.l2_mult * (2 * x + 1) / t**2
                )
            else:
                single_targets.append(self.lr)
                l2_targets.append(self.lr * self.l2_mult)
        targets = [single_targets[1:], l2_targets[1:]]
        optimizer = copy.deepcopy(self.optimizer)
        scheduler = QuadraticLRWarmup(optimizer, max_steps=t)
        _test_scheduler_value(optimizer, scheduler, targets, epochs)
