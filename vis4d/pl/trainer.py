# pylint: disable=consider-using-alias,consider-alternative-union-syntax
"""Vis4D Trainer."""
import os.path as osp
from datetime import datetime
from typing import List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.progress.base import ProgressBarBase
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.strategies import (  # type: ignore[attr-defined] # pylint: disable=line-too-long
    DDPStrategy,
)
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.utilities.device_parser import new_parse_gpu_ids

from vis4d.common import ArgsType
from vis4d.common.imports import TENSORBOARD_AVAILABLE, is_torch_tf32_available
from vis4d.common.logging import rank_zero_info, rank_zero_warn
from vis4d.pl.progress import DefaultProgressBar


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
        work_dir: str = "vis4d-workspace",
        exp_name: str = "unnamed",
        version: Optional[str] = None,
        find_unused_parameters: bool = False,
        checkpoint_period: int = 1,
        resume: bool = False,
        wandb: bool = False,
        tqdm: bool = False,
        use_tf32: bool = False,
        progress_bar_refresh_rate: int = 50,
        **kwargs: ArgsType,
    ) -> None:
        """Perform some basic common setups at the beginning of a job.

        1. Print environment info
        2. Setup callbacks: logger, LRMonitor, GPUMonitor, Checkpoint, etc
        3. Init distributed plugin
        """
        if is_torch_tf32_available():  # pragma: no cover
            if use_tf32:
                rank_zero_warn(
                    "Torch TF32 is available and turned on by default! "
                    + "It might harm the performance due to the precision. "
                    + "You can turn it off by setting trainer.use_tf32=False."
                )
            else:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False

        self.resume = resume
        self.work_dir = work_dir
        self.exp_name = exp_name
        if version is None:
            timestamp = (
                str(datetime.now())
                .split(".", maxsplit=1)[0]
                .replace(" ", "_")
                .replace(":", "-")
            )
            version = timestamp
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

        # add progress bar (train progress separate from validation)
        if tqdm:
            progress_bar: ProgressBarBase = TQDMProgressBar(
                progress_bar_refresh_rate
            )
        else:
            progress_bar = DefaultProgressBar(progress_bar_refresh_rate)
        callbacks += [progress_bar]

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

        # add distributed strategy
        if kwargs["accelerator"] == "gpu":  # pragma: no cover
            kwargs["devices"] = new_parse_gpu_ids(
                kwargs["devices"], include_cuda=True, include_mps=True
            )
            if len(kwargs["devices"]) > 1:
                strategy = kwargs["strategy"]
                if strategy == "ddp" or strategy is None:
                    ddp_plugin: Strategy = DDPStrategy(
                        find_unused_parameters=find_unused_parameters
                    )
                    kwargs["strategy"] = ddp_plugin
                else:
                    raise AttributeError(
                        f"Vis4D does not support strategy {strategy}"
                    )

        if "callbacks" not in kwargs or kwargs["callbacks"] is None:
            kwargs["callbacks"] = callbacks
        elif isinstance(kwargs["callbacks"], pl.callbacks.Callback):
            kwargs["callbacks"] = [kwargs["callbacks"], *callbacks]
        else:
            kwargs["callbacks"] += callbacks

        super().__init__(*args, **kwargs)

    @property
    def log_dir(self) -> Optional[str]:
        """Get current logging directory."""
        dirpath = self.strategy.broadcast(self.output_dir)
        return dirpath
