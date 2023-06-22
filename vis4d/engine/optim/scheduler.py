"""Vis4D LR schedulers."""
from __future__ import annotations

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
