"""Vis4D LR schedulers."""
from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class PolyLR(LRScheduler):
    """Polynomial learning rate decay.

    Example:
        Assuming lr = 0.001, max_steps = 4, and min_lr = 0.0, the learning rate
        will be:
        lr = 0.001     if step == 0
        lr = 0.00075   if step == 1
        lr = 0.00050   if step == 2
        lr = 0.00025   if step == 3
        lr = 0.0       if step >= 4

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_steps (int): Maximum number of steps.
        power (float, optional): Power factor. Default: 1.0.
        min_lr (float): Minimum learning rate. Default: 0.0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for each
            update. Default: ``False``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """Initialize PolyLRScheduler."""
        self.max_steps = max_steps
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:  # type: ignore
        """Compute current learning rate."""
        step_count = self._step_count  # type: ignore
        if step_count > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]
        coeff = (1 - (step_count - 1) / self.max_steps) ** self.power
        return [
            (base_lr - self.min_lr) * coeff + self.min_lr
            for base_lr in self.base_lrs
        ]


class YOLOXCosineAnnealingLR(LRScheduler):
    """YOLOX version of cosine annealing scheduler.

    Set the learning rate of each parameter group using a cosine annealing
    schedule and uses a fixed learning rate (eta_min) after the maximum number
    of iterations (max_steps).

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_steps (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(
        self, optimizer, max_steps, eta_min=0, last_epoch=-1, verbose=False
    ):
        self.max_steps = max_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        """Compute current learning rate."""
        if self._step_count == 1:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif self._step_count <= self.max_steps:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (
                    1
                    + math.cos(
                        (self._step_count - 1) * math.pi / self.max_steps
                    )
                )
                / 2
                for base_lr, group in zip(
                    self.base_lrs, self.optimizer.param_groups
                )
            ]
        return [self.eta_min for _ in self.optimizer.param_groups]
