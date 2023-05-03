"""Default runtime configuration for the project."""
import inspect
import platform
from datetime import datetime

import pytorch_lightning as pl

from vis4d.common.callbacks import CheckpointCallback, LoggingCallback
from vis4d.config.util import ConfigDict, class_config


def set_output_dir(config: ConfigDict) -> ConfigDict:
    """Set output directory for the experiment with timestamp."""
    timestamp = (
        str(datetime.now())
        .split(".", maxsplit=1)[0]
        .replace(" ", "_")
        .replace(":", "-")
    )
    config.timestamp = timestamp
    config.version = timestamp

    if platform.system() == "Windows":
        path_component = "\\"
    else:
        path_component = "/"

    config.output_dir = (
        config.get_ref("work_dir")
        + path_component
        + config.get_ref("experiment_name")
        + path_component
        + config.get_ref("version")
    )

    return config


def get_generic_callback_config(
    config: ConfigDict, params: ConfigDict
) -> tuple[dict[str, ConfigDict], dict[str, ConfigDict]]:
    """Get generic callback config.

    Here we define general, all purpose callbacks. Note, that these callbacks
    do not need to be registered with the data connector.
    """
    logger_callback = {
        "logger": class_config(LoggingCallback, refresh_rate=50)
    }
    ckpt_callback = {
        "ckpt": class_config(
            CheckpointCallback,
            save_prefix=config.output_dir,
            run_every_nth_epoch=1,
            num_epochs=params.num_epochs,
        )
    }

    return logger_callback, ckpt_callback


def get_pl_trainer_args() -> ConfigDict:
    """Get PyTorch Lightning Trainer arguments."""
    pl_trainer = ConfigDict()

    # PL Trainer arguments
    for k, v in inspect.signature(pl.Trainer).parameters.items():
        if not k in {"callbacks", "logger", "devices", "strategy"}:
            pl_trainer[k] = v.default

    # Default Trainer arguments
    pl_trainer.find_unused_parameters = False
    pl_trainer.checkpoint_period = 1
    pl_trainer.wandb = False

    return pl_trainer
