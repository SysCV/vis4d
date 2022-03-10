"""Vis4D Trainer."""
import itertools
import os.path as osp
from argparse import Namespace
from datetime import datetime
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import pandas
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin, DDPSpawnPlugin
from pytorch_lightning.utilities.cli import LightningCLI, SaveConfigCallback
from pytorch_lightning.utilities.device_parser import parse_gpu_ids
from pytorch_lightning.utilities.distributed import (
    rank_zero_info,
    rank_zero_warn,
)
from torch.utils import data
from torch.utils.collect_env import get_pretty_env_info

from ..common.registry import build_component
from ..config import Config, default_argument_parser, parse_config
from ..data.dataset import ScalabelDataset
from ..data.datasets import BaseDatasetLoader
from ..data.handler import BaseDatasetHandler
from ..data.module import BaseDataModule
from ..data.samplers import (
    BaseDistributedSampler,
    BaseSampler,
    build_data_sampler,
)
from ..model import BaseModel
from ..struct import DictStrAny, ModuleCfg
from ..vis import ScalabelWriterCallback
from .evaluator import StandardEvaluatorCallback
from .utils import Vis4DProgressBar, Vis4DTQDMProgressBar, setup_logger


class DefaultTrainer(pl.Trainer):
    """DefaultTrainer in Vis4D.

    Attributes:
        work_dir: Specific directory to save checkpoints, logs, etc. Integrates
        with exp_name and version to work_dir/exp_name/version.
        Default: ./vis4d-workspace/
        exp_name: Name of current experiment. Default: <name of model>
        version: Version of current experiment. Default: <timestamp>
        input_dir: Input directory in case you want to run inference on a folder
        with input data (e.g. images that can be temporally sorted by name).
        find_unused_parameters: Activates PyTorch checking for unused parameters
        in DDP setting. Deactivated by default for better performance.
        visualize: If you're running in predict mode, this option lets you
        visualize the model predictions in the output_dir.
        seed: Set random seed for numpy, torch, python. Default: None,
        i.e. no specific random seed is chosen.
        weights: Filepath for weights to load in test / predict. Default: "best",
        will load the best checkpoint in work_dir/exp_name/version.
        checkpoint_period: After N epochs, save out checkpoints. Default: 1
        resume: Whether to resume from weights (if specified), or last ckpt in
        work_dir/exp_name/version.
        wandb: Use weights and biases logging instead of tensorboard (default).
        not_strict: Whether to enforce keys in weights to be consistent with
        model's.
        tqdm: Activate tqdm based terminal logging behavior.
        legacy_ckpt: If model to load is a legacy checkpoint.
    """

    def __init__(
        self,
        *args,  # TODO check for overlap with existing args
        wandb: bool = False,
        tqdm: bool = False,
        work_dir: str = "vis4d-workspace",
        exp_name: str = "unnamed",
        version: str = (
            str(datetime.now())
            .split(".", maxsplit=1)[0]
            .replace(" ", "_")
            .replace(":", "-")
        ),
        input_dir: Optional[str] = None,
        find_unused_parameters: bool = False,
        visualize: bool = False,
        weights: Optional[str] = None,
        checkpoint_period: int = 1,
        resume: bool = False,
        not_strict: bool = False,
        legacy_ckpt: bool = False,
        tuner_params: Optional[DictStrAny] = None,
        tuner_metrics: Optional[List[str]] = None,
        **kwargs,
    ):  # TODO handle all params
        """Perform some basic common setups at the beginning of a job.

        2. Setup callbacks: logger, LRMonitor, GPUMonitor, Checkpoint
        3. Init pytorch lightning Trainer
        4. Set up cmd line logger
        5. Log information about environment, trainer arguments, etc
        6. Backup the args / config to the output directory
        """
        if input_dir is not None:
            if not os.path.exists(input_dir):
                raise FileNotFoundError(
                    f"Input directory does not exist: {input_dir}"
                )

        # print env, super init
        rank_zero_info("Environment info: %s", get_pretty_env_info())

        self.tuner_params = tuner_params
        self.tuner_metrics = tuner_metrics

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

        callbacks = []

        # add learning rate / GPU stats monitor (logs to tensorboard)
        callbacks += [
            pl.callbacks.LearningRateMonitor(logging_interval="step")
        ]

        # add progress bar (train progress separate from validation)
        if tqdm:
            progress_bar = Vis4DTQDMProgressBar()
        else:
            progress_bar = Vis4DProgressBar()
        callbacks += [progress_bar]

        # add Model checkpointer
        self.output_dir = osp.join(work_dir, exp_name, version)
        callbacks += [
            pl.callbacks.ModelCheckpoint(
                dirpath=osp.join(self.output_dir, "checkpoints"),
                verbose=True,
                save_last=True,
                every_n_epochs=checkpoint_period,
                save_on_train_epoch_end=True,
            )
        ]

        # resume from checkpoint if specified
        if resume and weights:  # pragma: no cover
            if osp.exists(osp.join(self.output_dir, "checkpoints/last.ckpt")):
                weights = osp.join(self.output_dir, "checkpoints/last.ckpt")
            else:
                raise ValueError(
                    "resume set to True but there is no checkpoint to "
                    "resume! Please specify a checkpoint via weights "
                    "or configure a directory that contains a checkpoint at "
                    "work_dir/exp_name/version/checkpoints/last.ckpt."
                )
        # trainer_args["ckpt_path"] = weights TODO move to cli

        # add distributed plugin
        if "gpus" in kwargs:  # pragma: no cover
            gpu_ids = parse_gpu_ids(kwargs["gpus"])
            num_gpus = len(gpu_ids) if gpu_ids is not None else 0
            if num_gpus > 1:
                if (
                    kwargs["accelerator"] == "ddp"
                    or kwargs["accelerator"] is None
                ):
                    ddp_plugin = DDPPlugin(
                        find_unused_parameters=find_unused_parameters
                    )
                    kwargs["plugins"] = [ddp_plugin]
                elif kwargs["accelerator"] == "ddp_spawn":
                    ddp_plugin = DDPSpawnPlugin(
                        find_unused_parameters=find_unused_parameters
                    )  # type: ignore
                    kwargs["plugins"] = [ddp_plugin]
                elif kwargs["accelerator"] == "ddp2":
                    ddp_plugin = DDP2Plugin(
                        find_unused_parameters=find_unused_parameters
                    )
                    kwargs["plugins"] = [ddp_plugin]
                if (
                    cfg.data is not None
                    and "train_sampler" in cfg.data
                    and cfg.data["train_sampler"] is not None
                    and training
                ):
                    # using custom sampler
                    kwargs["replace_sampler_ddp"] = False

        if "callbacks" not in kwargs or kwargs["callbacks"] is None:
            kwargs["callbacks"] = callbacks
        elif isinstance(kwargs["callbacks"], pl.callbacks.Callback):
            kwargs["callbacks"] = [kwargs["callbacks"], *callbacks]
        else:
            kwargs["callbacks"] += callbacks

        # setup cmd line logging, print and save info about trainer
        setup_logger(osp.join(self.output_dir, "log.txt"))
        super().__init__(*args, **kwargs)

    @property
    def log_dir(self) -> Optional[str]:
        dirpath = self.accelerator.broadcast(self.output_dir)
        return dirpath

    def tune(self) -> None:
        """Tune function."""
        rank_zero_info("Starting hyperparameter search...")
        if self.tuner_params is None:
            raise ValueError(
                "Tuner parameters not defined! Please specify "
                "tuner_params in Trainer arguments."
            )
        if self.tuner_metrics is None:
            raise ValueError(
                "Tuner metrics not defined! Please specify "
                "tuner_metrics in Trainer arguments."
            )
        search_params = self.tuner_params
        search_metrics = self.tuner_metrics
        param_names = list(search_params.keys())
        param_groups = list(product(*search_params.values()))
        metrics_all = {}
        for param_group in param_groups:
            for key, value in zip(param_names, param_group):
                obj = model
                for name in key.split(".")[:-1]:
                    obj = getattr(obj, name, None)  # type: ignore
                    if obj is None:
                        raise ValueError(
                            f"Attribute {name} not found in {key}!"
                        )
                setattr(obj, key.split(".")[-1], value)

            metrics = self.test(verbose=False)
            if len(metrics) > 0:
                rank_zero_warn(
                    "More than one dataloader found, but tuning "
                    "requires a single dataset to tune parameters on!"
                )
            metrics_all[str(param_group)] = {
                k: v for k, v in metrics[0].items() if k in search_metrics
            }
        rank_zero_info("Done!")
        rank_zero_info("The following parameters have been considered:")
        rank_zero_info("\n" + str(list(search_params.keys())))
        rank_zero_info("Hyperparameter search result:")
        rank_zero_info("\n" + str(pandas.DataFrame.from_dict(metrics_all)))


def setup_category_mapping(
    data_cfgs: List[ModuleCfg],
    model_category_mapping: Optional[Dict[str, int]],
) -> None:
    """Setup category_mapping for each dataset."""
    for data_cfg in data_cfgs:
        if (
            "category_mapping" not in data_cfg
            or data_cfg["category_mapping"] is None
        ):
            if model_category_mapping is not None:
                # default to using model category_mapping, if exists
                data_cfg["category_mapping"] = model_category_mapping
            continue


class BaseCLI(LightningCLI):
    """Default CLI for Vis4D"""

    def __init__(
        self,
        model_class: Optional[
            Union[Type[BaseModel], Callable[..., BaseModel]]
        ] = None,
        datamodule_class: Optional[
            Union[Type[BaseDataModule], Callable[..., BaseDataModule]]
        ] = None,
        save_config_callback: Optional[
            Type[SaveConfigCallback]
        ] = SaveConfigCallback,
        trainer_class: Union[
            Type[pl.Trainer], Callable[..., pl.Trainer]
        ] = DefaultTrainer,
        seed_everything_default: Optional[int] = None,
        description: str = "Vis4D command line tool",
        env_prefix: str = "V4D",
        **kwargs: ArgsType,
    ) -> None:
        if seed_everything_default is not None:
            rank_zero_info("Using random seed: %s", seed_everything_default)
        super().__init__(
            model_class=model_class,
            datamodule_class=datamodule_class,
            save_config_callback=save_config_callback,
            trainer_class=trainer_class,
            description=description,
            env_prefix=env_prefix,
            seed_everything_default=seed_everything_default,
            **kwargs,
        )

    def instantiate_classes(self) -> None:
        super().instantiate_classes()
        self.datamodule.set_cat_map(self.model.category_mapping)
