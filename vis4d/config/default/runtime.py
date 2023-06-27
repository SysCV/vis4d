"""Default runtime configuration for the project."""
import platform
from datetime import datetime

from ml_collections import ConfigDict

from vis4d.config import FieldConfigDict, class_config
from vis4d.config.common.types import ExperimentConfig
from vis4d.engine.callbacks import CheckpointCallback, LoggingCallback


def get_default_cfg(
    exp_name: str, work_dir: str = "vis4d-workspace"
) -> ExperimentConfig:
    """Set default config for the project.

    It will set the following fields:
        - work_dir (str): Default to "vis4d-workspace"
        - experiment_name (str): Experiment name.
        - timestamp (str): Current time
        - version (str): Same as timestamp
        - output_dir (str): work_dir/experiment_name/version

    Args:
        exp_name (str): Experiment name.
        work_dir (str, optional): Working directory. Defaults to
            "vis4d-workspace".

    Returns:
        FieldConfigDict: Config for the project.
    """
    config = FieldConfigDict()

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
    config: FieldConfigDict, refresh_rate: int = 50
) -> list[ConfigDict]:
    """Get default callbacks config.

    It will return a list of callbacks config including:
        - LoggingCallback
        - CheckpointCallback

    Args:
        config (FieldConfigDict): Config for the project.
        refresh_rate (int, optional): Refresh rate for the logging. Defaults to
            50.

    Returns:
        list[ConfigDict]: List of callbacks config.
    """
    callbacks = []

    # Logger
    callbacks.append(class_config(LoggingCallback, refresh_rate=refresh_rate))

    # Checkpoint
    callbacks.append(
        class_config(CheckpointCallback, save_prefix=config.output_dir)
    )

    return callbacks
