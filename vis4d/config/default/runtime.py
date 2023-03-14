"""Default runtime configuration for the project."""
from datetime import datetime

from vis4d.config.util import ConfigDict


def set_output_dir(config: ConfigDict) -> ConfigDict:
    """Set output directory for the experiment with timestamp."""
    timestamp = (
        str(datetime.now())
        .split(".", maxsplit=1)[0]
        .replace(" ", "_")
        .replace(":", "-")
    )
    config.version = timestamp
    config.timestamp = timestamp

    config.output_dir = (
        config.get_ref("work_dir")
        + "/"
        + config.get_ref("experiment_name")
        + "/"
        + config.get_ref("version")
    )
    return config
