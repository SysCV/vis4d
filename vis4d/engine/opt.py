"""Vis4D optimizer."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Protocol

from torch import nn, optim

from vis4d.optim.warmup import BaseLRWarmup


class OptimizerBuilder(Protocol):
    """Protocol for optimizer builder."""

    # not sure why this is necessary by mypy complains missing __name__ without
    __name__: str = "OptimizerBuilder"

    def __call__(self, params: Iterator[nn.Parameter]) -> optim.Optimizer:
        """Returns the optimizer for the desired parameters.

        Args:
            params: The parameters that will be optimized.

        Returns:
            The optimizer.
        """


class LRSchedulerBuilder(Protocol):
    """Protocol for LR scheduler builder."""

    __name__: str = "LRSchedulerBuilder"

    def __call__(
        self, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler._LRScheduler:
        """Returns the scheduler for the desired optimizer.

        Args:
            optimizer: The optimizer.

        Returns:
            The LR Scheduler.
        """


class Optimizer:
    """Vis4D Optimizer.

    This class is responsible for creating the optimizer and learning rate
    scheduler. It also handles the learning rate warmup.
    """

    def __init__(
        self,
        optimizer_cb: OptimizerBuilder,
        lr_scheduler_cb: LRSchedulerBuilder | None = None,
        lr_warmup: None | BaseLRWarmup = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            optimizer_cb: A callback that creates the optimizer with the
                desired parameters.
            lr_scheduler_cb: A callback that creates the learning rate
                scheduler.
            lr_warmup: The learning rate warmup.
        """
        self._warmup = lr_warmup
        self._optimizer_cb = optimizer_cb
        self._lr_scheduler_cb = lr_scheduler_cb

        # These need to be set in setup() since they might depend on the model
        self.lr_scheduler: optim.lr_scheduler._LRScheduler | None = None
        self.optimizer: None | optim.Optimizer = None

    def setup(self, model: nn.Module) -> None:
        """Setup optimizer.

        Args:
            model: The model with the corresponding weights that will
                be optimized.
        This creates the optimizer and the learning rate scheduler.
        Note that this needs to be called before zero_grad() and step().
        """
        self.optimizer = self._optimizer_cb(params=model.parameters())
        self.lr_scheduler = (
            self._lr_scheduler_cb(optimizer=self.optimizer)
            if self._lr_scheduler_cb is not None
            else None
        )

    def zero_grad(self) -> None:
        """Zero gradients in optimizer."""
        assert self.optimizer is not None, (
            "Optimizer was not correctly setup. Make sure to call setup()"
            "before zero_grad()."
        )
        self.optimizer.zero_grad()

    def step(
        self, step: int, closure: Callable[[], float] | None = None
    ) -> None:
        """Step optimizer.

        This function will first step the optimizer, then the warmup or the
        learning rate scheduler.
        Note that the learning rate scheduler will only be stepped if the
        warmup is finished.

        This function should be called after zero_grad().

        Args:
            step: The current step of the training loop.
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        Raises:
            ValueError: If the base learning rate could not be determined.
        """
        assert self.optimizer is not None, (
            "Optimizer was not correctly setup. Make sure to call setup()"
            "before step()."
        )

        # Optimizer step
        self.optimizer.step(closure=closure)
        # Warmup step
        warmed_up = self.warmup_step(step)
        # LR scheduler step
        if self.lr_scheduler is not None and not warmed_up:
            self.lr_scheduler.step()

    def warmup_step(self, step: int) -> bool:
        """Set learning rate according to warmup.

        Args:
            step: The current step of the training loop.

        Raises:
            ValueError: If the base learning rate could not be determined.

        Returns:
            True if the warmup is finished, False otherwise.
        """
        assert self.optimizer is not None, "Optimizer was not correctly setup."

        base_lr = self.optimizer.defaults.get("lr", None)
        if base_lr is None:
            raise ValueError(
                "Couldn't determine base LR from optimizer defaults: "
                f"{self.optimizer.defaults}"
            )

        if self._warmup is not None and step <= self._warmup.warmup_steps:
            print(self._warmup(step, base_lr))
            for g in self.optimizer.param_groups:
                if step < self._warmup.warmup_steps:
                    g["lr"] = self._warmup(step, base_lr)
                else:
                    g["lr"] = base_lr

        return self._warmup is not None and step <= self._warmup.warmup_steps
