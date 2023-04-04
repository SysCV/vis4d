"""Example to show connected components in the config."""
from vis4d.config.data_graph import prints_datagraph_for_config

from absl import app
from vis4d.engine.parser import DEFINE_config_file

from vis4d.config.util import instantiate_classes

_CONFIG = DEFINE_config_file("config", method_name="get_config")


def main(argv) -> None:  # type:ignore
    """Main entry point for the CLI."""
    config = _CONFIG.value
    dg = prints_datagraph_for_config(instantiate_classes(config))
    print(dg)


if __name__ == "__main__":
    app.run(main)
