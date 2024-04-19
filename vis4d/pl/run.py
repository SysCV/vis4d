"""CLI interface using PyTorch Lightning."""

from __future__ import annotations

import logging
import os.path as osp

import torch
from absl import app
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch import Callback
from torch.utils.collect_env import get_pretty_env_info

from vis4d.common import ArgsType
from vis4d.common.logging import dump_config, rank_zero_info, setup_logger
from vis4d.common.util import set_tf32
from vis4d.config import instantiate_classes
from vis4d.config.typing import ExperimentConfig
from vis4d.engine.callbacks import CheckpointCallback
from vis4d.engine.flag import _CKPT, _CONFIG, _GPUS, _RESUME, _SHOW_CONFIG
from vis4d.engine.parser import pprints_config
from vis4d.pl.callbacks import CallbackWrapper, LRSchedulerCallback
from vis4d.pl.data_module import DataModule
from vis4d.pl.trainer import PLTrainer
from vis4d.pl.training_module import TrainingModule


def main(argv: ArgsType) -> None:
    """Main entry point for the CLI.

    Example to run this script:
    >>> python -m vis4d.pl.run fit --config configs/faster_rcnn/faster_rcnn_coco.py
    """
    # Get config
    mode = argv[1]
    assert mode in {"fit", "test"}, f"Invalid mode: {mode}"
    config: ExperimentConfig = _CONFIG.value
    num_gpus = _GPUS.value

    # Setup logging
    logger_vis4d = logging.getLogger("vis4d")
    logger_pl = logging.getLogger("pytorch_lightning")
    log_file = osp.join(config.output_dir, f"log_{config.timestamp}.txt")
    setup_logger(logger_vis4d, log_file)
    setup_logger(logger_pl, log_file)

    # Dump config
    config_file = osp.join(
        config.output_dir, f"config_{config.timestamp}.yaml"
    )
    dump_config(config, config_file)

    rank_zero_info("Environment info: %s", get_pretty_env_info())

    # PyTorch Setting
    set_tf32(config.use_tf32, config.tf32_matmul_precision)
    torch.hub.set_dir(f"{config.work_dir}/.cache/torch/hub")

    # Setup device
    if num_gpus > 0:
        config.pl_trainer.accelerator = "gpu"
        config.pl_trainer.devices = num_gpus
    else:
        config.pl_trainer.accelerator = "cpu"
        config.pl_trainer.devices = 1

    trainer_args = instantiate_classes(config.pl_trainer).to_dict()

    if _SHOW_CONFIG.value:
        rank_zero_info(pprints_config(config))

    # Instantiate classes
    if mode == "fit":
        train_data_connector = instantiate_classes(config.train_data_connector)
        loss = instantiate_classes(config.loss)
    else:
        train_data_connector = None
        loss = None

    if config.test_data_connector is not None:
        test_data_connector = instantiate_classes(config.test_data_connector)
    else:
        test_data_connector = None

    # Callbacks
    callbacks: list[Callback] = []
    for cb in config.callbacks:
        callback = instantiate_classes(cb)
        # Skip checkpoint callback to use PL ModelCheckpoint
        if not isinstance(callback, CheckpointCallback):
            callbacks.append(CallbackWrapper(callback))

    if "pl_callbacks" in config:
        pl_callbacks = [instantiate_classes(cb) for cb in config.pl_callbacks]
    else:
        pl_callbacks = []

    for cb in pl_callbacks:
        if not isinstance(cb, Callback):
            raise MisconfigurationException(
                "Callback must be a subclass of pytorch_lightning Callback. "
                f"Provided callback: {cb} is not!"
            )
        callbacks.append(cb)

    # Add needed callbacks
    callbacks.append(LRSchedulerCallback())

    # Checkpoint path
    ckpt_path = _CKPT.value

    # Resume training
    resume = _RESUME.value
    if resume:
        if ckpt_path is None:
            resume_ckpt_path = osp.join(
                config.output_dir, "checkpoints/last.ckpt"
            )
        else:
            resume_ckpt_path = ckpt_path
    else:
        resume_ckpt_path = None

    trainer = PLTrainer(callbacks=callbacks, **trainer_args)

    hyper_params = trainer_args

    if config.get("params", None) is not None:
        hyper_params.update(config.params.to_dict())

    training_module = TrainingModule(
        config.model,
        config.optimizers,
        loss,
        train_data_connector,
        test_data_connector,
        hyper_params,
        config.seed,
        ckpt_path if not resume else None,
    )
    data_module = DataModule(config.data)

    if mode == "fit":
        trainer.fit(
            training_module, datamodule=data_module, ckpt_path=resume_ckpt_path
        )
    elif mode == "test":
        trainer.test(training_module, datamodule=data_module, verbose=False)


def entrypoint() -> None:
    """Entry point for the CLI."""
    app.run(main)


if __name__ == "__main__":
    entrypoint()
