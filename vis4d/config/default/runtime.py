"""Default runtime configuration for the project."""
import platform
from datetime import datetime

from vis4d.engine.callbacks import CheckpointCallback, LoggingCallback
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


def get_callback_config(
    config: ConfigDict, params: ConfigDict, refresh_rate: int = 50
) -> list[ConfigDict]:
    """Get default callback config."""
    callbacks = []

    # Logger
    callbacks.append(class_config(LoggingCallback, refresh_rate=refresh_rate))

    # Checkpoint
    callbacks.append(
        class_config(CheckpointCallback, save_prefix=config.output_dir)
    )

    return callbacks
