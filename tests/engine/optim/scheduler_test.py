"""Test cases for learning rate schedulers."""
import math

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import LRScheduler
from torch.testing import assert_close

from vis4d.engine.optim.scheduler import YOLOXCosineAnnealingLR


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


def test_yolox_cos_anneal_scheduler():
    """Test case for YOLOXCosineAnnealingLR."""
    model = ToyModel()
    lr = 0.05
    layer2_mult = 10
    optimizer = optim.SGD(
        [
            {"params": model.conv1.parameters()},
            {"params": model.conv2.parameters(), "lr": lr * layer2_mult},
        ],
        lr=lr,
        momentum=0.01,
        weight_decay=5e-4,
    )
    epochs = 12
    t = 10
    eta_min = 1e-10
    single_targets = [
        eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / t)) / 2
        if x < t
        else eta_min
        for x in range(epochs)
    ]
    targets = [
        single_targets,
        [
            x * layer2_mult if i < t else eta_min
            for i, x in enumerate(single_targets)
        ],
    ]
    scheduler = YOLOXCosineAnnealingLR(optimizer, max_steps=t, eta_min=eta_min)
    _test_scheduler_value(optimizer, scheduler, targets, epochs)
