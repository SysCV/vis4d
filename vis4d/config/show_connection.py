"""Example to show connected components in the config."""
from absl import app

from vis4d.config.data_graph import prints_datagraph_for_config
from vis4d.config.util import instantiate_classes
from vis4d.engine.parser import DEFINE_config_file

_CONFIG = DEFINE_config_file("config", method_name="get_config")


def main(argv) -> None:  # type:ignore # pylint: disable=unused-argument
    """Main entry point to show connected components in the config.

    >>> python -m vis4d.config.show_connection --config configs/faster_rcnn/faster_rcnn_coco.py
    """
    config = _CONFIG.value
    dg = prints_datagraph_for_config(instantiate_classes(config))
    print(dg)


if __name__ == "__main__":
    app.run(main)
