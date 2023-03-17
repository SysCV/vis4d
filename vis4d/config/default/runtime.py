"""Default runtime configuration for the project."""
from datetime import datetime

from vis4d.config.util import ConfigDict


def get_runtime_config(config: ConfigDict) -> ConfigDict:
    """Returns the default runtime configuration for the project."""
    timestamp = (
        str(datetime.now())
        .split(".", maxsplit=1)[0]
        .replace(" ", "_")
        .replace(":", "-")
    )
    config.timestamp = timestamp
    config.version = timestamp

    config.output_dir = (
        config.get_ref("work_dir")
        + "/"
        + config.get_ref("experiment_name")
        + "/"
        + config.get_ref("version")
    )
    return config
