"""CLI interface for vis4d.

Example to run this script:
>>> python -m vis4d.pl.cli --config vis4d/config/example/faster_rcnn_coco.py
"""
from __future__ import annotations

from typing import Any

from absl import app, flags
from ml_collections import ConfigDict
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from vis4d.common.logging import rank_zero_info
from vis4d.config.util import instantiate_classes, pprints_config
from vis4d.engine.parser import DEFINE_config_file
from vis4d.pl.callbacks.callback_wrapper import CallbackWrapper
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


def get_default(config: ConfigDict, key: str, default) -> Any:
    return config[key] if key in config else default


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

    if _GPUS.value != config.n_gpus:
        # Replace # gpus in config with cli
        config.n_gpus = _GPUS.value  # patch it like this to update potential
        # references to the config
    num_gpus = config.n_gpus

    if _SHOW_CONFIG.value:
        rank_zero_info("*" * 80)
        rank_zero_info(pprints_config(config))
        rank_zero_info("*" * 80)

    # Load Trainer kwargs from config
    cfg: ConfigDict = instantiate_classes(config)

    pl_config = get_default(cfg, "pl", ConfigDict())
    trainer_args = get_default(pl_config, "trainer", ConfigDict())
    pl_callbacks = get_default(pl_config, "callbacks", [])

    # Update GPU mode
    if num_gpus > 0:
        trainer_args.devices = num_gpus
        trainer_args.accelerator = "gpu"

    callbacks: list[Callback] = []
    if "train_callbacks" in cfg:
        for key, cb in cfg.train_callbacks.items():
            rank_zero_info(f"Adding callback {key}")
            callbacks.append(CallbackWrapper(cb, cfg.data_connector, key))

    if "test_callbacks" in cfg:
        for key, cb in cfg.test_callbacks.items():
            rank_zero_info(f"Adding callback {key}")

            callbacks.append(CallbackWrapper(cb, cfg.data_connector, key))

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

    if _MODE.value == "train":
        trainer.fit(
            TrainingModule(
                cfg.model, cfg.optimizers, cfg.loss, cfg.data_connector
            ),
            cfg.train_dl,
        )
    elif _MODE.value == "test":
        trainer.test(
            TrainingModule(
                cfg.model, cfg.optimizers, cfg.loss, cfg.data_connector
            ),
            cfg.train_dl,
        )


if __name__ == "__main__":
    app.run(main)
