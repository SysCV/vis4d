"""Vis4D LR schedulers."""
from __future__ import annotations

from typing import TypedDict

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from vis4d.common.typing import DictStrAny
from vis4d.config import instantiate_classes
from vis4d.config.typing import LrSchedulerConfig


class LRSchedulerDict(TypedDict):
    """LR scheduler."""

    scheduler: LRScheduler
    begin: int
    end: int
    epoch_based: bool


class LRSchedulerWrapper(LRScheduler):
    """LR scheduler wrapper."""

    def __init__(
        self, lr_schedulers_cfg: list[LrSchedulerConfig], optimizer: Optimizer
    ) -> None:
        """Initialize LRSchedulerWrapper."""
        self.lr_schedulers_cfg = lr_schedulers_cfg
        self.lr_schedulers: dict[int, LRSchedulerDict] = {}
        super().__init__(optimizer)

        for i, lr_scheduler_cfg in enumerate(self.lr_schedulers_cfg):
            if lr_scheduler_cfg["begin"] == 0:
                self._instantiate_lr_scheduler(i, lr_scheduler_cfg)

    def _instantiate_lr_scheduler(
        self, scheduler_idx: int, lr_scheduler_cfg: LrSchedulerConfig
    ) -> None:
        """Instantiate LR schedulers."""
        # OneCycleLR needs max_lr to be set
        if "max_lr" in lr_scheduler_cfg["scheduler"]["init_args"]:
            lr_scheduler_cfg["init_args"]["max_lrs"] = [
                pg["lr"] for pg in self.optimizer.param_groups
            ]

        self.lr_schedulers[scheduler_idx] = {
            "scheduler": instantiate_classes(
                lr_scheduler_cfg["scheduler"], optimizer=self.optimizer
            ),
            "begin": lr_scheduler_cfg["begin"],
            "end": lr_scheduler_cfg["end"],
            "epoch_based": lr_scheduler_cfg["epoch_based"],
        }

    def get_lr(self) -> list[float]:  # type: ignore
        """Get current learning rate."""
        return [
            lr_scheduler["scheduler"].get_lr()
            for lr_scheduler in self.lr_schedulers.values()
        ]

    def state_dict(self) -> dict[int, DictStrAny]:  # type: ignore
        """Get state dict."""
        state_dict = {}
        for scheduler_idx, lr_scheduler in self.lr_schedulers.items():
            state_dict[scheduler_idx] = lr_scheduler["scheduler"].state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[int, DictStrAny]) -> None:  # type: ignore # pylint: disable=line-too-long
        """Load state dict."""
        for scheduler_idx, _state_dict in state_dict.items():
            # Instantiate the lr scheduler if it is not instantiated yet
            if not scheduler_idx in self.lr_schedulers:
                self._instantiate_lr_scheduler(
                    scheduler_idx, self.lr_schedulers_cfg[scheduler_idx]
                )
            self.lr_schedulers[scheduler_idx]["scheduler"].load_state_dict(
                _state_dict
            )

    def _step_lr(self, lr_scheduler: LRSchedulerDict, step: int) -> None:
        """Step the learning rate."""
        if lr_scheduler["begin"] <= step and (
            lr_scheduler["end"] == -1 or lr_scheduler["end"] > step
        ):
            lr_scheduler["scheduler"].step()

    def step(self, epoch: int | None = None) -> None:
        """Step on training epoch end."""
        if epoch is not None:
            for lr_scheduler in self.lr_schedulers.values():
                if lr_scheduler["epoch_based"]:
                    self._step_lr(lr_scheduler, epoch)

            for i, lr_scheduler_cfg in enumerate(self.lr_schedulers_cfg):
                if lr_scheduler_cfg["epoch_based"] and (
                    lr_scheduler_cfg["begin"] == epoch + 1
                ):
                    self._instantiate_lr_scheduler(i, lr_scheduler_cfg)

    def step_on_batch(self, step: int) -> None:
        """Step on training batch end."""
        # Minus 1 because the step is called after the optimizer.step()
        step -= 1
        for lr_scheduler in self.lr_schedulers.values():
            if not lr_scheduler["epoch_based"]:
                self._step_lr(lr_scheduler, step)

        for i, lr_scheduler_cfg in enumerate(self.lr_schedulers_cfg):
            if not lr_scheduler_cfg["epoch_based"] and (
                lr_scheduler_cfg["begin"] == step + 1
            ):
                self._instantiate_lr_scheduler(i, lr_scheduler_cfg)


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
