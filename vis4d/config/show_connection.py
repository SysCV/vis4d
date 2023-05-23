"""Example to show connected components in the config."""
from absl import app

from vis4d.common import ArgsType
from vis4d.config.data_graph import prints_datagraph_for_config
from vis4d.config.parser import DEFINE_config_file
from vis4d.config.util import instantiate_classes

_CONFIG = DEFINE_config_file("config", method_name="get_config")


def main(
    argv: ArgsType,  # pylint: disable=unused-argument
) -> None:  # pragma: no cover
    """Main entry point to show connected components in the config.

    >>> python -m vis4d.config.show_connection --config configs/faster_rcnn/faster_rcnn_coco.py
    """
    config = _CONFIG.value

    train_data_connector = instantiate_classes(config.train_data_connector)
    test_data_connector = instantiate_classes(config.test_data_connector)
    loss = instantiate_classes(config.loss)
    model = instantiate_classes(config.model)
    callbacks = instantiate_classes(config.callbacks)

    dg = prints_datagraph_for_config(
        model, train_data_connector, test_data_connector, loss, callbacks
    )
    print(dg)


if __name__ == "__main__":  # pragma: no cover
    app.run(main)
