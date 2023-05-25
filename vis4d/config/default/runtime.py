"""Default runtime configuration for the project."""
import platform
from datetime import datetime

from vis4d.config import ConfigDict, class_config
from vis4d.engine.callbacks import CheckpointCallback, LoggingCallback


def get_default_cfg(
    exp_name: str, work_dir: str = "vis4d-workspace"
) -> ConfigDict:
    """Set default config for the project."""
    config = ConfigDict()

    config.work_dir = work_dir
    config.experiment_name = exp_name

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


def get_default_callbacks_cfg(
    config: ConfigDict, refresh_rate: int = 50
) -> list[ConfigDict]:
    """Get default callbacks config."""
    callbacks = []

    # Logger
    callbacks.append(class_config(LoggingCallback, refresh_rate=refresh_rate))

    # Checkpoint
    callbacks.append(
        class_config(CheckpointCallback, save_prefix=config.output_dir)
    )

    return callbacks
