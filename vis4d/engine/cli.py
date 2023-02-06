"""CLI interface for vis4d.

Example to run this script:
>>> python -m vis4d.engine.cli --config vis4d/config/example/faster_rcnn_coco.py
"""
# Functional Interface
from __future__ import annotations

import logging  # TODO change to vis4d logging
import sys

import yaml
from absl import app, flags
from ml_collections import ConfigDict
from ml_collections.config_flags.config_flags import (
    _ConfigFileParser,
    _ConfigFlag,
)

from vis4d.config.replicator import replicate_config
from vis4d.config.util import instantiate_classes, pprints_config
from vis4d.engine.test import Tester
from vis4d.engine.train import Trainer


## Parser Setup
class ConfigFileParser(_ConfigFileParser):  # type: ignore
    """Parser for config files.

    Note, this wraps internal functions of the ml_collections code and might
    be fragile!
    """

    def parse(self, path: str) -> ConfigDict:
        """Returns the config object for a given path.

        If a colon is present in `path`, everything to the right of the first
        colon is passed to `get_config` as an argument. This allows the
        structure of what  is returned to be modified.

        Works with .py file that contain a get_config() function and .yaml.

        Args:
          path (string): path pointing to the config file to execute. May also
              contain a config_string argument, e.g. be of the form
              "config.py:some_configuration" or "config.yaml".
        Returns (ConfigDict):
          ConfigDict located at 'path'
        """
        if path.split(".")[-1] == "yaml":
            with open(path, "r", encoding="utf-8") as yaml_file:
                data_dict = ConfigDict(yaml.safe_load(yaml_file))

                if self._lock_config:
                    data_dict.lock()
                return data_dict
        else:
            return super().parse(path)

    def flag_type(self) -> str:
        """The flag type of this object.

        Returns:
            str: config object
        """
        return "config object"


def DEFINE_config_file(  # pylint: disable=invalid-name
    name: str,
    default: str | None = None,
    help_string: str = "path to config file [.py |.yaml].",
    lock_config: bool = True,
) -> flags.FlagHolder:
    """Registers a new flag for a config file.

    Args:
        name (str): The name of the flag (e.g. config for --config flag)
        default (str | None, optional): Default Value. Defaults to None.
        help_string (str, optional): Help String.
            Defaults to "path to config file.".
        lock_config (bool, optional): Whether or note to lock the returned
            config. Defaults to True.

    Returns:
        flags.FlagHolder: Flag holder instance.
    """
    parser = ConfigFileParser(name=name, lock_config=lock_config)
    flag = _ConfigFlag(
        parser=parser,
        serializer=flags.ArgumentSerializer(),
        name=name,
        default=default,
        help_string=help_string,
        flag_values=flags.FLAGS,
    )

    # Get the module name for the frame at depth 1 in the call stack.
    module_name = sys._getframe(  # pylint: disable=protected-access
        1
    ).f_globals.get("__name__", None)
    module_name = sys.argv[0] if module_name == "__main__" else module_name
    return flags.DEFINE_flag(flag, flags.FLAGS, module_name=module_name)


## Done Parser Setup

# TODO: Currently this does not allow to load multpile config files.
# extend functionality to chain multiple config files using
# e.g. --config=model_1.py --config=loader_args.py
# or --config=my_config.py --config.train_dl=different_dl.py
_CONFIG = DEFINE_config_file("config")
_SWEEP = DEFINE_config_file("sweep")
_MODE = flags.DEFINE_string(
    "mode", default="train", help="Choice of [train, test]"
)

logger = logging.getLogger(__name__)


def _train(config: ConfigDict) -> None:
    """Train the model."""
    # TODO, connect this to SLURM cluster to directly spawn jobs.
    logger.info("Starting training")

    logger.info("*" * 80)
    logger.info(pprints_config(config))
    logger.info("*" * 80)

    cfg: ConfigDict = instantiate_classes(config)
    trainer = Trainer(
        num_epochs=cfg.engine.num_epochs,
        log_step=1,
        dataloaders=cfg.train_dl,
        data_connector=cfg.data_connector,
        train_callbacks=cfg.get("train_callbacks", None),
    )
    tester = Tester(
        dataloaders=cfg.test_dl,
        data_connector=cfg.data_connector,
        test_callbacks=cfg.get("test_callbacks", None),
    )

    # run training
    trainer.train(
        cfg.model, cfg.optimizers, cfg.loss, tester, cfg.engine.metric
    )


def main(  # type:ignore # pylint: disable=unused-argument
    *args, **kwargs
) -> None:
    """Main entry point for the CLI.

    Example to run this script:
    >>> python -m vis4d.engine.cli --config vis4d/config/example/faster_rcnn_coco.py

    Or to run a parameter sweep:
    >>> python -m vis4d.engine.cli --config vis4d/config/example/faster_rcnn_coco.py --sweep vis4d/config/example/lr_sweep.py
    """
    if _SWEEP.value is not None:
        # Perform parameter sweep
        # TODO, improve. Where to save the results? What name for the run?
        config = _CONFIG.value
        sweep_obj = instantiate_classes(_SWEEP.value)

        for config in replicate_config(
            _CONFIG.value,
            method=sweep_obj.method,
            sampling_args=sweep_obj.sampling_args,
        ):
            _train(config)
    elif _MODE.value == "train":
        _train(_CONFIG.value)


if __name__ == "__main__":
    app.run(main)
