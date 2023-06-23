"""CLI interface using PyTorch Lightning."""
from __future__ import annotations

import logging
import os.path as osp

from absl import app, flags
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch import Callback
from torch.utils.collect_env import get_pretty_env_info

from vis4d.common import ArgsType
from vis4d.common.logging import rank_zero_info, setup_logger
from vis4d.common.util import set_tf32
from vis4d.config import instantiate_classes
from vis4d.engine.callbacks.checkpoint import CheckpointCallback
from vis4d.engine.parser import DEFINE_config_file, pprints_config
from vis4d.pl.callbacks import CallbackWrapper, LRWarmUpCallback
from vis4d.pl.data_module import DataModule
from vis4d.pl.trainer import PLTrainer
from vis4d.pl.training_module import TrainingModule

# TODO: Support resume from folder and load config directly from it.
_CONFIG = DEFINE_config_file("config", method_name="get_config")
_GPUS = flags.DEFINE_integer("gpus", default=0, help="Number of GPUs")
_CKPT = flags.DEFINE_string("ckpt", default=None, help="Checkpoint path")
_RESUME = flags.DEFINE_bool("resume", default=False, help="Resume training")
_SHOW_CONFIG = flags.DEFINE_bool(
    "print-config", default=False, help="If set, prints the configuration."
)


def main(argv: ArgsType) -> None:
    """Main entry point for the CLI.

    Example to run this script:
    >>> python -m vis4d.pl.cli fit --config configs/faster_rcnn/faster_rcnn_coco.py
    """
    # Get config
    mode = argv[1]
    assert mode in {"fit", "test"}, f"Invalid mode: {mode}"
    config = _CONFIG.value
    num_gpus = _GPUS.value

    # Setup logging
    logger_vis4d = logging.getLogger("vis4d")
    logger_pl = logging.getLogger("pytorch_lightning")
    log_file = osp.join(config.output_dir, f"log_{config.timestamp}.txt")
    setup_logger(logger_vis4d, log_file)
    setup_logger(logger_pl, log_file)

    rank_zero_info("Environment info: %s", get_pretty_env_info())

    # PyTorch Setting
    set_tf32(False)

    # Setup GPU
    config.pl_trainer.devices = num_gpus
    if num_gpus > 0:
        config.pl_trainer.accelerator = "gpu"

    trainer_args = instantiate_classes(config.pl_trainer).to_dict()

    # TODO: Add random seed and DDP
    if _SHOW_CONFIG.value:
        rank_zero_info(pprints_config(config))

    # Seed
    seed = config.get("seed", None)

    # Instantiate classes
    if mode == "fit":
        train_data_connector = instantiate_classes(config.train_data_connector)
        loss = instantiate_classes(config.loss)
    else:
        train_data_connector = None
        loss = None

    test_data_connector = instantiate_classes(config.test_data_connector)

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
    callbacks.append(LRWarmUpCallback())

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
    training_module = TrainingModule(
        config.model,
        config.optimizers,
        loss,
        train_data_connector,
        test_data_connector,
        {**config.params.to_dict(), **trainer_args},
        seed,
        ckpt_path if not resume else None,
    )
    data_module = DataModule(config.data)

    if mode == "fit":
        trainer.fit(
            training_module, datamodule=data_module, ckpt_path=resume_ckpt_path
        )
    elif mode == "test":
        trainer.test(training_module, datamodule=data_module, verbose=False)


if __name__ == "__main__":
    app.run(main)
