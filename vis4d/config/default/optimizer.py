"""Default optimizer configuration."""
from __future__ import annotations

from ml_collections import ConfigDict

from vis4d.config.util import class_config, delay_instantiation
from vis4d.engine.opt import Optimizer


def get_optimizer_config(
    optimizer: ConfigDict,
    lr_scheduler: ConfigDict | None = None,
    lr_warmup: ConfigDict | None = None,
    epoch_based_lr: bool = False,
    epoch_based_warmup: bool = False,
) -> ConfigDict:
    """Default optimizer configuration.

    This creates a config object that can be initialized as an Optimizer for
    training. It takes the optimizer and learning rate scheduler as callbacks
    that are called during setup.

    Args:
        optimizer (ConfigDict): Optimizer configuration.
        lr_scheduler (ConfigDict, optional): Learning rate scheduler
        lr_warmup (ConfigDict, optional): Learning rate warmup.
        epoch_based_lr (bool, optional): Whether the learning rate scheduler is
            epoch based or step based. Defaults to False.
        epoch_based_warmup (bool, optional): Whether the warmup is epoch based
            or step based. Defaults to False.

    Returns:
        ConfigDict: Config dict that can be instantiated as Optimizer.

    Example:
    >>> # Set up optimizer config
    >>>torch_optim = class_config(SGD, lr=0.01, momentum=0.9)
    >>> # Set up learning rate scheduler config
    >>> lr_scheduler_cfg = class_config(SingleCycleLR, max_lr=0.01, ...)
    >>> # Set up learning rate warmup config
    >>> lr_warmup_cfg = class_config(LinearWarmup, warmup_steps=1000, ...)
    >>> # get default optimizer
    >>> cfg = optimizer_cfg(torch_optim, lr_scheduler_cfg, lr_warmup_cfg)
    """
    return class_config(
        Optimizer,
        optimizer_cb=delay_instantiation(instantiable=optimizer),
        lr_scheduler_cb=delay_instantiation(instantiable=lr_scheduler)
        if lr_scheduler is not None
        else None,
        lr_warmup=lr_warmup,
        epoch_based_lr=epoch_based_lr,
        epoch_based_warmup=epoch_based_warmup,
    )
