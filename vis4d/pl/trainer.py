# pylint: disable=consider-using-alias,consider-alternative-union-syntax
"""Vis4D Trainer."""
from __future__ import annotations

import os.path as osp
from typing import List, Optional

import pytorch_lightning as pl
from pytorch_lightning.strategies import (  # type: ignore[attr-defined] # pylint: disable=line-too-long
    DDPStrategy,
)
from pytorch_lightning.strategies.strategy import Strategy

from vis4d.common import ArgsType
from vis4d.common.imports import TENSORBOARD_AVAILABLE
from vis4d.common.logging import rank_zero_info
from vis4d.common.util import set_tf32


class DefaultTrainer(pl.Trainer):
    """DefaultTrainer in Vis4D.

    Attributes:
        work_dir: Specific directory to save checkpoints, logs, etc. Integrates
        with exp_name and version to work_dir/exp_name/version.
        Default: ./vis4d-workspace/
        exp_name: Name of current experiment. Default: unnamed
        version: Version of current experiment. Default: <timestamp>
        find_unused_parameters: Activates PyTorch checking for unused
        parameters in DDP setting. Default: False, for better performance.
        checkpoint_period: After N epochs, save out checkpoints. Default: 1
        resume: Whether to resume from weights (if specified), or last ckpt in
        work_dir/exp_name/version.
        wandb: Use weights and biases logging instead of tensorboard (default).
        not_strict: Whether to enforce keys in weights to be consistent with
        model's.
        tqdm: Activate tqdm based terminal logging behavior.
        tuner_params: which parameters to tune.
    """

    def __init__(
        self,
        *args: ArgsType,
        work_dir: str,
        exp_name: str,
        version: str,
        find_unused_parameters: bool = False,
        checkpoint_period: int = 1,
        resume: bool = False,
        wandb: bool = False,
        use_tf32: bool = False,
        **kwargs: ArgsType,
    ) -> None:
        """Perform some basic common setups at the beginning of a job.

        1. Setup env.
        2. Setup logger.
        2. Setup callbacks.
        3. Init distributed plugin
        """
        set_tf32(use_tf32)

        self.resume = resume
        self.work_dir = work_dir
        self.exp_name = exp_name
        self.version = version

        self.output_dir = osp.join(work_dir, exp_name, version)

        # setup experiment logging
        if "logger" not in kwargs or (
            isinstance(kwargs["logger"], bool) and kwargs["logger"]
        ):
            if wandb:  # pragma: no cover
                exp_logger = pl.loggers.WandbLogger(  # type: ignore[attr-defined] # pylint: disable=line-too-long
                    save_dir=work_dir,
                    project=exp_name,
                    name=version,
                )
            elif TENSORBOARD_AVAILABLE:
                exp_logger = pl.loggers.TensorBoardLogger(
                    save_dir=work_dir,
                    name=exp_name,
                    version=version,
                    default_hp_metric=False,
                )
            else:
                exp_logger = None
                rank_zero_info(
                    "Neither `tensorboard` nor `tensorboardX` is "
                    "available. Running without experiment logger. To log "
                    "your experiments, try `pip install`ing either."
                )
            kwargs["logger"] = exp_logger

        callbacks: List[pl.callbacks.Callback] = []

        # add learning rate / GPU stats monitor (logs to tensorboard)
        if TENSORBOARD_AVAILABLE or wandb:
            callbacks += [
                pl.callbacks.LearningRateMonitor(logging_interval="step")
            ]

        # add Model checkpointer
        callbacks += [
            pl.callbacks.ModelCheckpoint(
                dirpath=osp.join(self.output_dir, "checkpoints"),
                verbose=True,
                save_last=True,
                every_n_epochs=checkpoint_period,
                save_on_train_epoch_end=True,
            )
        ]

        kwargs["callbacks"] += callbacks

        # add distributed strategy
        if (
            kwargs["accelerator"] == "gpu" and kwargs["devices"] > 1
        ):  # pragma: no cover
            ddp_plugin: Strategy = DDPStrategy(
                find_unused_parameters=find_unused_parameters
            )
            kwargs["strategy"] = ddp_plugin

        super().__init__(*args, **kwargs)

    @property
    def log_dir(self) -> Optional[str]:
        """Get current logging directory."""
        dirpath = self.strategy.broadcast(self.output_dir)
        return dirpath
