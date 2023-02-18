"""CLI interface for vis4d.

Example to run this script:
>>> python -m vis4d.pl.cli --config vis4d/config/example/faster_rcnn_coco.py
"""
from __future__ import annotations

from absl import app, flags
from ml_collections import ConfigDict
from pytorch_lightning import Trainer

from vis4d.common.logging import rank_zero_info
from vis4d.config.util import instantiate_classes, pprints_config
from vis4d.engine.parser import DEFINE_config_file
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
    if "trainer_kwargs" in cfg:
        trainer_kwargs = cfg.trainer_kwargs
    else:
        trainer_kwargs = {}

    # Update GPU mode
    if num_gpus > 0:
        trainer_kwargs["devices"] = num_gpus
        trainer_kwargs["accelerator"] = "gpu"

    trainer = Trainer(**trainer_kwargs)

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
