"""Vis4D LR schedulers."""
from __future__ import annotations

from typing import TypedDict

from ml_collections import ConfigDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from vis4d.common.typing import DictStrAny
from vis4d.config import instantiate_classes


class LRSchedulerDict(TypedDict):
    """LR scheduler."""

    scheduler: LRScheduler
    begin: int
    end: int
    epoch_based: bool


class LRSchedulerWrapper(LRScheduler):
    """LR scheduler wrapper."""

    def __init__(
        self, lr_schedulers_cfg: ConfigDict, optimizer: Optimizer
    ) -> None:
        """Initialize LRSchedulerWrapper."""
        self.lr_schedulers_cfg = lr_schedulers_cfg
        self.lr_schedulers: list[LRSchedulerDict] = []
        super().__init__(optimizer)

        for lr_scheduler_cfg in self.lr_schedulers_cfg:
            if lr_scheduler_cfg["begin"] == 0:
                self._instantiate_lr_scheduler(lr_scheduler_cfg)

    def _instantiate_lr_scheduler(self, lr_scheduler_cfg: ConfigDict) -> None:
        """Instantiate LR schedulers."""
        # OneCycleLR needs max_lr to be set
        if "max_lr" in lr_scheduler_cfg["scheduler"]["init_args"]:
            lr_scheduler_cfg["init_args"]["max_lrs"] = [
                pg["lr"] for pg in self.optimizer.param_groups
            ]

        self.lr_schedulers.append(
            {
                "scheduler": instantiate_classes(
                    lr_scheduler_cfg["scheduler"], optimizer=self.optimizer
                ),
                "begin": lr_scheduler_cfg["begin"],
                "end": lr_scheduler_cfg["end"],
                "epoch_based": lr_scheduler_cfg["epoch_based"],
            }
        )

    def get_lr(self) -> list[float]:  # type: ignore
        """Get current learning rate."""
        return [
            lr_scheduler["scheduler"].get_lr()
            for lr_scheduler in self.lr_schedulers
        ]

    def state_dict(self) -> list[DictStrAny]:  # type: ignore
        """Get state dict."""
        return [
            lr_scheduler["scheduler"].state_dict()
            for lr_scheduler in self.lr_schedulers
        ]

    def load_state_dict(self, state_dict: list[DictStrAny]) -> None:  # type: ignore # pylint: disable=line-too-long
        """Load state dict."""
        for lr_scheduler, state_dict_ in zip(self.lr_schedulers, state_dict):
            lr_scheduler["scheduler"].load_state_dict(state_dict_)

    def _step_lr(self, lr_scheduler: LRSchedulerDict, step: int) -> None:
        """Step the learning rate."""
        if lr_scheduler["begin"] <= step and (
            lr_scheduler["end"] == -1 or lr_scheduler["end"] > step
        ):
            lr_scheduler["scheduler"].step()

    def step(self, epoch: int | None = None) -> None:
        """Step on training epoch end."""
        if epoch is not None:
            for lr_scheduler in self.lr_schedulers:
                if lr_scheduler["epoch_based"]:
                    self._step_lr(lr_scheduler, epoch)

            for lr_scheduler_cfg in self.lr_schedulers_cfg:
                if lr_scheduler_cfg["epoch_based"] and (
                    lr_scheduler_cfg["begin"] == epoch + 1
                ):
                    self._instantiate_lr_scheduler(lr_scheduler_cfg)

    def step_on_batch(self, step: int) -> None:
        """Step on training batch end."""
        # Minus 1 because the step is called after the optimizer.step()
        step -= 1
        for lr_scheduler in self.lr_schedulers:
            if not lr_scheduler["epoch_based"]:
                self._step_lr(lr_scheduler, step)

        for lr_scheduler_cfg in self.lr_schedulers_cfg:
            if not lr_scheduler_cfg["epoch_based"] and (
                lr_scheduler_cfg["begin"] == step + 1
            ):
                self._instantiate_lr_scheduler(lr_scheduler_cfg)


class ConstantLR(LRScheduler):
    """Constant learning rate scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_steps (int): Maximum number of steps.
        factor (float): Scale factor. Default: 1.0 / 3.0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for each
            update. Default: ``False``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        factor: float = 1.0 / 3.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """Initialize ConstantLR."""
        self.max_steps = max_steps
        self.factor = factor
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:  # type: ignore
        """Compute current learning rate."""
        step_count = self._step_count - 1  # type: ignore
        if step_count == 0:
            return [
                group["lr"] * self.factor
                for group in self.optimizer.param_groups
            ]
        if step_count == self.max_steps:
            return [
                group["lr"] * (1.0 / self.factor)
                for group in self.optimizer.param_groups
            ]
        return [group["lr"] for group in self.optimizer.param_groups]


class PolyLR(LRScheduler):
    """Polynomial learning rate decay.

    Example:
        Assuming lr = 0.001, max_steps = 4, min_lr = 0.0, and power = 1.0, the
        learning rate will be:
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
        step_count = self._step_count - 1  # type: ignore
        if step_count == 0 or step_count > self.max_steps:
            return [group["lr"] for group in self.optimizer.param_groups]
        decay_factor = (
            (1.0 - step_count / self.max_steps)
            / (1.0 - (step_count - 1) / self.max_steps)
        ) ** self.power
        return [
            (group["lr"] - self.min_lr) * decay_factor + self.min_lr
            for group in self.optimizer.param_groups
        ]


class QuadraticLRWarmup(LRScheduler):
    """Quadratic learning rate warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_steps (int): Maximum number of steps.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for each
            update. Default: ``False``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """Initialize QuadraticLRWarmup."""
        self.max_steps = max_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:  # type: ignore
        """Compute current learning rate."""
        step_count = self._step_count - 1  # type: ignore
        if step_count >= self.max_steps:
            return self.base_lrs
        factors = [
            base_lr * (2 * step_count + 1) / self.max_steps**2
            for base_lr in self.base_lrs
        ]
        if step_count == 0:
            return factors
        return [
            group["lr"] + factor
            for factor, group in zip(factors, self.optimizer.param_groups)
        ]
