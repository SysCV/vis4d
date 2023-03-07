"""CLI interface for vis4d.

Example to run this script:
>>> python -m vis4d.pl.cli --config vis4d/config/example/faster_rcnn_coco.py
"""
from __future__ import annotations

import logging
import os.path as osp

from absl import app, flags
from ml_collections import ConfigDict
from pytorch_lightning import Callback
from pytorch_lightning.utilities.exceptions import (  # type: ignore[attr-defined] # pylint: disable=line-too-long
    MisconfigurationException,
)
from torch.utils.collect_env import get_pretty_env_info

from vis4d.common.logging import rank_zero_info, setup_logger
from vis4d.config.util import instantiate_classes, pprints_config
from vis4d.engine.parser import DEFINE_config_file
from vis4d.pl.callbacks.callback_wrapper import CallbackWrapper
from vis4d.pl.data_module import DataModule
from vis4d.pl.trainer import DefaultTrainer
from vis4d.pl.training_module import TrainingModule

_CONFIG = DEFINE_config_file("config", method_name="get_config")
_MODE = flags.DEFINE_string(
    "mode", default="train", help="Choice of [train, test]"
)
_GPUS = flags.DEFINE_integer("gpus", default=0, help="Number of GPUs")
_SHOW_CONFIG = flags.DEFINE_bool(
    "print-config", default=False, help="If set, prints the configuration."
)


def main(  # type:ignore # pylint: disable=unused-argument
    *args, **kwargs
) -> None:
    """Main entry point for the CLI.

    Example to run this script:
    >>> python -m vis4d.engine.cli --config vis4d/config/example/faster_rcnn_coco.py

    Or to run a parameter sweep:
    >>> python -m vis4d.engine.cli --config vis4d/config/example/faster_rcnn_coco.py --sweep vis4d/config/example/faster_rcnn_coco.py
    """
    config = _CONFIG.value
    config.n_gpus = _GPUS.value

    # Setup logging
    logger_vis4d = logging.getLogger("vis4d")
    logger_pl = logging.getLogger("pytorch_lightning")
    log_file = osp.join(config.output_dir, f"log_{config.timestamp}.txt")
    setup_logger(logger_vis4d, log_file)
    setup_logger(logger_pl, log_file)

    rank_zero_info("Environment info: %s", get_pretty_env_info())

    if _SHOW_CONFIG.value:
        rank_zero_info("*" * 80)
        rank_zero_info(pprints_config(config))
        rank_zero_info("*" * 80)

    # Load Trainer kwargs from config
    trainer_args_cfg = ConfigDict()
    pl_trainer = instantiate_classes(config.pl_trainer)
    for key, value in pl_trainer.items():
        trainer_args_cfg[key] = value
    trainer_args_cfg.max_epochs = config.params.num_epochs
    trainer_args_cfg.num_sanity_val_steps = 0

    # Update GPU mode
    if config.n_gpus > 0:
        trainer_args_cfg.devices = config.n_gpus
        trainer_args_cfg.accelerator = "gpu"

    # Disable progress bar for logger
    trainer_args_cfg.enable_progress_bar = False
    trainer_args_cfg.work_dir = config.work_dir
    trainer_args_cfg.exp_name = config.experiment_name
    trainer_args_cfg.version = config.version

    trainer_args = instantiate_classes(trainer_args_cfg)

    # Instantiate classes
    data_connector = instantiate_classes(config.data_connector)
    model = instantiate_classes(config.model)
    optimizers = instantiate_classes(config.optimizers)
    loss = instantiate_classes(config.loss)

    # Callbacks
    callbacks: list[Callback] = []
    if "shared_callbacks" in config:
        shared_callbacks = instantiate_classes(config.shared_callbacks)
        for key, cb in shared_callbacks.items():
            rank_zero_info(f"Adding callback {key}")
            callbacks.append(CallbackWrapper(cb, data_connector, key))

    if "train_callbacks" in config and _MODE.value == "train":
        train_callbacks = instantiate_classes(config.train_callbacks)
        for key, cb in train_callbacks.items():
            rank_zero_info(f"Adding callback {key}")
            callbacks.append(CallbackWrapper(cb, data_connector, key))

    if "test_callbacks" in config:
        test_callbacks = instantiate_classes(config.test_callbacks)
        for key, cb in test_callbacks.items():
            rank_zero_info(f"Adding callback {key}")
            callbacks.append(CallbackWrapper(cb, data_connector, key))

    if "pl_callbacks" in config:
        pl_callbacks = instantiate_classes(config.pl_callbacks)
    else:
        pl_callbacks = []

    for cb in pl_callbacks:
        if not isinstance(cb, Callback):
            raise MisconfigurationException(
                "Callback must be a subclass of "
                "pytorch_lightning.Callback. Provided "
                f"callback: {cb} is not a subclass of "
                "pytorch_lightning.Callback."
            )

        callbacks.append(cb)

    trainer = DefaultTrainer(callbacks=callbacks, **trainer_args)
    data_module = DataModule(config.data)

    if _MODE.value == "train":
        trainer.fit(
            TrainingModule(model, optimizers, loss, data_connector),
            datamodule=data_module,
        )
    elif _MODE.value == "test":
        trainer.test(
            TrainingModule(model, optimizers, loss, data_connector),
            datamodule=data_module,
        )


if __name__ == "__main__":
    app.run(main)
