"""Vis4D optimizer."""
from __future__ import annotations

from collections.abc import Callable

from ml_collections import ConfigDict
from torch import nn, optim

from vis4d.common.logging import rank_zero_info
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
        optimizer = configure_optimizer(model, optim_cfg)
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


def configure_optimizer(
    model: nn.Module, optim_cfg: ConfigDict
) -> optim.Optimizer:
    """Configure optimizer with parameter groups."""
    base_lr = optim_cfg.optimizer["init_args"].lr
    weight_decay = optim_cfg.optimizer["init_args"].get("weight_decay", None)

    param_groups_cfg = optim_cfg.get("param_groups_cfg", None)
    params = []

    # One cycle lr
    one_cycle = "max_lr" in optim_cfg.lr_scheduler["init_args"]

    if param_groups_cfg is not None:
        for group in param_groups_cfg:
            lr_mult = group.get("lr_mult", 1.0)
            decay_mult = group.get("decay_mult", 1.0)

            param_group = {"params": [], "lr": base_lr * lr_mult}

            if weight_decay is not None:
                param_group["weight_decay"] = weight_decay * decay_mult

            params.append(param_group)

        # Create a param group for the rest of the parameters
        param_group = {"params": [], "lr": base_lr}
        if weight_decay is not None:
            param_group["weight_decay"] = weight_decay
        params.append(param_group)

        # Add the parameters to the param groups
        add_params(params, model, param_groups_cfg)

        if one_cycle:
            max_lrs = [pg.pop("lr") for pg in params]
            optim_cfg.lr_scheduler["init_args"]["max_lr"] = max_lrs
    else:
        params = model.parameters()

    return instantiate_classes(optim_cfg.optimizer, params=params)


def add_params(
    params: list[dict],
    module: nn.Module,
    param_groups_cfg: dict[str, list[str] | float],
    prefix: str = "",
) -> None:
    """Add all parameters of module to the params list.

    The parameters of the given module will be added to the list of param
    groups, with specific rules defined by paramwise_cfg.

    Args:
        params (list[dict]): A list of param groups, it will be modified
            in place.
        module (nn.Module): The module to be added.
        param_groups_cfg (dict[str, list[str] | float]): The configuration
            of the param groups.
        prefix (str): The prefix of the module. Default: ''.
    """
    for name, param in module.named_parameters(recurse=False):
        if not param.requires_grad:
            params[-1]["params"].append(param)
            continue

        # if the parameter match one of the custom keys, ignore other rules
        for i, group in enumerate(param_groups_cfg):
            for key in group["custom_keys"]:
                if key in f"{prefix}.{name}":
                    rank_zero_info(
                        f"{prefix}.{name} with lr_multi: {group['lr_mult']}"
                    )
                    params[i]["params"].append(param)
                    break

    for child_name, child_mod in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        add_params(params, child_mod, param_groups_cfg, prefix=child_prefix)
