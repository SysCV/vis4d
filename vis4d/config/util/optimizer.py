"""Optimizer configuration."""
from __future__ import annotations

from ml_collections import ConfigDict


def get_optimizer_cfg(
    optimizer: ConfigDict,
    lr_scheduler: ConfigDict | None = None,
    lr_warmup: ConfigDict | None = None,
    epoch_based_lr: bool = True,
    epoch_based_warmup: bool = False,
) -> ConfigDict:
    """Default optimizer configuration.

    This creates a config object that can be initialized as an Optimizer for
    training.

    Args:
        optimizer (ConfigDict): Optimizer configuration.
        lr_scheduler (ConfigDict, optional): Learning rate scheduler. Defaults
            to None.
        lr_warmup (ConfigDict, optional): Learning rate warmup. Defaults to
            None.
        epoch_based_lr (bool, optional): Whether the learning rate scheduler is
            epoch based or step based. Defaults to True.
        epoch_based_warmup (bool, optional): Whether the warmup is epoch based
            or step based. Defaults to False.

    Returns:
        ConfigDict: Config dict that can be instantiated as Optimizer.
    """
    optim = ConfigDict()

    optim.optimizer = optimizer
    optim.lr_scheduler = lr_scheduler
    optim.lr_warmup = lr_warmup
    optim.epoch_based_lr = epoch_based_lr
    optim.epoch_based_warmup = epoch_based_warmup

    return optim
