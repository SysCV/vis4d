"""Vis4D LR schedulers."""
from __future__ import annotations

from torch.optim import Optimizer, lr_scheduler


class PolyLRScheduler(
    lr_scheduler._LRScheduler  # pylint: disable=protected-access
):
    """Polynomial learning rate decay."""

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

    def get_lr(self) -> list[float]:
        """Compute current learning rate."""
        if self._step_count >= self.max_steps:  # pragma: no cover
            return [self.min_lr for _ in self.base_lrs]
        coeff = (1 - self._step_count / self.max_steps) ** self.power
        return [
            (base_lr - self.min_lr) * coeff + self.min_lr
            for base_lr in self.base_lrs
        ]
