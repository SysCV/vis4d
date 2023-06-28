"""Optimizer configuration."""
from __future__ import annotations

from ml_collections import ConfigDict

from vis4d.config.common.types import LrSchedulerConfig, OptimizerConfig
from vis4d.engine.optim import ParamGroupsCfg


def get_lr_scheduler_cfg(
    scheduler: ConfigDict,
    begin: int = 0,
    end: int = -1,
    epoch_based: bool = True,
) -> LrSchedulerConfig:
    """Default learning rate scheduler configuration.

    This creates a config object that can be initialized as a LearningRate
    scheduler for training.

    Args:
        scheduler (ConfigDict): Learning rate scheduler configuration.
        begin (int, optional): Begin epoch. Defaults to 0.
        end (int, optional): End epoch. Defaults to None. Defaults to -1.
        epoch_based (bool, optional): Whether the learning rate scheduler is
            epoch based or step based. Defaults to True.

    Returns:
        LrSchedulerConfig: Config dict that can be instantiated as LearningRate
            scheduler.
    """
    lr_scheduler = LrSchedulerConfig()

    lr_scheduler.scheduler = scheduler
    lr_scheduler.begin = begin
    lr_scheduler.end = end
    lr_scheduler.epoch_based = epoch_based

    return lr_scheduler


def get_optimizer_cfg(
    optimizer: ConfigDict,
    lr_schedulers: list[LrSchedulerConfig] | None = None,
    param_groups: list[ParamGroupsCfg] | None = None,
) -> OptimizerConfig:
    """Default optimizer configuration.

    This creates a config object that can be initialized as an Optimizer for
    training.

    Args:
        optimizer (ConfigDict): Optimizer configuration.
        lr_schedulers (list[LrSchedulerConfig] | None, optional): Learning rate
            schedulers configuration. Defaults to None.
        param_groups (list[ParamGroupsCfg] | None, optional): Parameter groups
            configuration. Defaults to None.

    Returns:
        OptimizerConfig: Config dict that can be instantiated as Optimizer.
    """
    optim = OptimizerConfig()

    optim.optimizer = optimizer
    optim.lr_schedulers = lr_schedulers
    optim.param_groups = param_groups

    return optim
