"""Vis4D Trainer."""
import os.path as osp
from datetime import datetime
from typing import Callable, List, Optional, Type, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.progress.base import ProgressBarBase
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.core import LightningModule
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin, DDPSpawnPlugin
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.utilities.cli import LightningCLI, SaveConfigCallback
from pytorch_lightning.utilities.device_parser import parse_gpu_ids
from pytorch_lightning.utilities.rank_zero import (
    rank_zero_info,
    rank_zero_warn,
)
from torch.utils.collect_env import get_pretty_env_info

from ..common import ArgsType, DictStrAny
from .data import DataModule
from .utils import DefaultProgressBar, is_torch_tf32_available, setup_logger


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
        tuner_metrics: which metrics to observe while tuning.
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
        tuner_params: Optional[DictStrAny] = None,
        tuner_metrics: Optional[List[str]] = None,
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

        self.tuner_params = tuner_params
        self.tuner_metrics = tuner_metrics
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
                exp_logger = pl.loggers.WandbLogger(
                    save_dir=work_dir,
                    project=exp_name,
                    name=version,
                )
            else:
                exp_logger = pl.loggers.TensorBoardLogger(  # type: ignore
                    save_dir=work_dir,
                    name=exp_name,
                    version=version,
                    default_hp_metric=False,
                    log_graph=True,
                )
            kwargs["logger"] = exp_logger

        callbacks: List[pl.callbacks.Callback] = []

        # add learning rate / GPU stats monitor (logs to tensorboard)
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

        # add distributed plugin
        if "gpus" in kwargs:  # pragma: no cover
            gpu_ids = parse_gpu_ids(
                kwargs["gpus"], include_cuda=True, include_mps=True
            )
            num_gpus = len(gpu_ids) if gpu_ids is not None else 0
            if num_gpus > 1:
                if kwargs["strategy"] == "ddp" or kwargs["strategy"] is None:
                    ddp_plugin: Strategy = DDPPlugin(
                        find_unused_parameters=find_unused_parameters
                    )
                    kwargs["plugins"] = [ddp_plugin]
                elif kwargs["strategy"] == "ddp_spawn":
                    ddp_plugin = DDPSpawnPlugin(
                        find_unused_parameters=find_unused_parameters
                    )
                    kwargs["plugins"] = [ddp_plugin]
                elif kwargs["strategy"] == "ddp2":
                    ddp_plugin = DDP2Plugin(
                        find_unused_parameters=find_unused_parameters
                    )
                    kwargs["plugins"] = [ddp_plugin]

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
        return dirpath  # type: ignore


class CLI(LightningCLI):
    """Basic pytorch lightning CLI in Vis4D."""

    def __init__(  # type: ignore
        self,
        model_class: Optional[
            Union[Type[LightningModule], Callable[..., LightningModule]]
        ] = None,
        datamodule_class: Optional[
            Union[Type[DataModule], Callable[..., DataModule]]
        ] = None,
        save_config_callback: Optional[
            Type[SaveConfigCallback]
        ] = SaveConfigCallback,
        trainer_class: Union[
            Type[pl.Trainer], Callable[..., pl.Trainer]
        ] = DefaultTrainer,
        description: str = "Vis4D command line tool",
        env_prefix: str = "V4D",
        save_config_overwrite: bool = True,
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(
            model_class=model_class,
            datamodule_class=datamodule_class,
            save_config_callback=save_config_callback,
            trainer_class=trainer_class,
            description=description,
            env_prefix=env_prefix,
            save_config_overwrite=save_config_overwrite,
            **kwargs,
        )

    def instantiate_classes(self) -> None:
        """Instantiate trainer, datamodule and model."""
        # setup cmd line logging, print env info
        subcommand = self.config["subcommand"]
        work_dir = self.config[subcommand].trainer.work_dir
        exp_name = self.config[subcommand].trainer.exp_name
        version = self.config[subcommand].trainer.version
        timestamp = (
            str(datetime.now())
            .split(".", maxsplit=1)[0]
            .replace(" ", "_")
            .replace(":", "-")
        )
        if version is None:
            version = timestamp
        self.config[subcommand].trainer.version = version
        setup_logger(
            osp.join(work_dir, exp_name, version, f"log_{timestamp}.txt")
        )
        rank_zero_info("Environment info: %s", get_pretty_env_info())

        # instantiate classes
        # self.config[subcommand].data.subcommand = subcommand
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._get(self.config_init, "data")
        self.model = self._get(self.config_init, "model")
        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()
        assert isinstance(self.trainer, DefaultTrainer), (
            "Trainer needs to inherit from DefaultTrainer "
            "for BaseCLI to work properly."
        )

        if self.trainer.resume:  # pragma: no cover
            weights = self.config_init[subcommand].ckpt_path
            if weights is None:
                weights = osp.join(
                    self.trainer.output_dir, "checkpoints/last.ckpt"
                )
            self.config_init[subcommand].ckpt_path = weights
