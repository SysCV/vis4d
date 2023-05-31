"""Vis4D optimizer."""
from __future__ import annotations

from collections.abc import Callable

from ml_collections import ConfigDict
from torch import nn, optim

from vis4d.config import instantiate_classes

from .warmup import BaseLRWarmup


class Optimizer:
    """Optimizer class.

    It is responsible for creating the optimizer and learning rate scheduler.
    It also handles the learning rate warmup.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler | None = None,
        lr_warmup: BaseLRWarmup | None = None,
        epoch_based_lr: bool = False,
        epoch_based_warmup: bool = False,
    ) -> None:
        """Creates an instance of the class.

        Args:
            optimizer (optim.Optimizer): The optimizer.
            lr_scheduler (optim.lr_scheduler._LRScheduler, optional): The
                learning rate scheduler. Defaults to None.
            lr_warmup (BaseLRWarmup, optional): The learning rate
                warmup. Defaults to None.
            epoch_based_lr (bool): Whether the learning rate scheduler
                should be based on epochs or batches. If True, the learning
                rate scheduler will be conducted per epoch. If
                False, the learning rate scheduler will be
                conducted per batch. Defaults to False.
            epoch_based_warmup (bool): Whether the warmup should be based on
                epochs or batches. If True, the warmup will be conducted per
                epoch. If False, the warmup will be conducted per batch.
                Defaults to False.
        """
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self._warmup = lr_warmup
        self.epoch_based_lr = epoch_based_lr
        self.epoch_based_warmup = epoch_based_warmup

        if self._warmup is not None:
            _ = self._warmup_step(0)

    def zero_grad(self) -> None:
        """Zero gradients in optimizer."""
        assert self.optimizer is not None, (
            "Optimizer was not correctly setup. Make sure to call setup()"
            "before zero_grad()."
        )
        self.optimizer.zero_grad()

    def step_on_batch(
        self, step: int, closure: Callable[[], float] | None = None
    ) -> None:
        """Step optimizer on batch end.

        This function will first step the learning rate scheduler or the warmup
        on batch end, then call the optimizer step. Note that this function
        should be called after zero_grad() of previous batch. Note that the
        learning rate scheduler will only be stepped if the warmup is finished.

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
        self.optimizer.step(closure=closure)

        # Adjust learning rate for next step
        if self.epoch_based_warmup:
            warmed_up = True
        else:
            warmed_up = self._warmup_step(step + 1)
        if not self.epoch_based_lr and warmed_up:
            self._lr_step()

    def step_on_epoch(self, epoch: int) -> None:
        """Step optimizer on epoch end.

        This function is used to step the learning rate scheduler or the warmup
        on epoch end. Note that the learning rate scheduler will only
        be stepped if the warmup is finished.

        Args:
            epoch: The current epoch of the training loop.

        Raises:
            ValueError: If the base learning rate could not be determined.
        """
        assert self.optimizer is not None, (
            "Optimizer was not correctly setup. Make sure to call setup()"
            "before step()."
        )
        if self.epoch_based_warmup:
            warmed_up = self._warmup_step(epoch + 1)
        else:
            warmed_up = True
        if self.epoch_based_lr and warmed_up:
            self._lr_step()

    def _lr_step(self) -> None:
        """Step learning rate scheduler.

        Raises:
            ValueError: If the base learning rate could not be determined.
        """
        assert self.optimizer is not None, (
            "Optimizer was not correctly setup. Make sure to call setup()"
            "before step()."
        )
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _warmup_step(self, step: int) -> bool:
        """Set learning rate according to warmup.

        Args:
            step: The current step of the training loop.

        Raises:
            ValueError: If the base learning rate could not be determined.

        Returns:
            True if the warmup is finished, False otherwise.
        """
        assert self.optimizer is not None, "Optimizer was not correctly setup."

        if self._warmup is not None and step <= self._warmup.warmup_steps:
            for g in self.optimizer.param_groups:
                if step < self._warmup.warmup_steps:
                    g["lr"] = self._warmup(step, g["initial_lr"])
                else:
                    g["lr"] = g["initial_lr"]
            return False
        return True


def set_up_optimizers(
    optimizers_cfg: list[ConfigDict], model: nn.Module
) -> list[Optimizer]:
    """Set up optimizers."""
    optimizers = []
    for optim_cfg in optimizers_cfg:
        optimizer = instantiate_classes(
            optim_cfg.optimizer, params=model.parameters()
        )
        lr_scheduler = (
            instantiate_classes(optim_cfg.lr_scheduler, optimizer=optimizer)
            if optim_cfg.lr_scheduler is not None
            else None
        )
        lr_warmup = (
            instantiate_classes(optim_cfg.lr_warmup)
            if optim_cfg.lr_warmup is not None
            else None
        )
        optimizers.append(
            Optimizer(
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                lr_warmup=lr_warmup,
                epoch_based_lr=optim_cfg.epoch_based_lr,
                epoch_based_warmup=optim_cfg.epoch_based_warmup,
            )
        )
    return optimizers
