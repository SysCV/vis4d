"""Show config connection tests."""
import unittest

from tests.zoo.util import content_equal, get_config_for_name
from vis4d.config import instantiate_classes
from vis4d.config.show_connection import prints_datagraph_for_config


class TestShowConfig(unittest.TestCase):
    """Tests the content of the provided configs for Show."""

    gt_config_path = "tests/vis4d-test-data/config_test"

    def test_show_frcnn(self) -> None:
        """Test the config for faster_rcnn_coco.py.

        This instantiates the config and compares it to a ground truth.
        """
        config = get_config_for_name("faster_rcnn.faster_rcnn_coco")

        train_data_connector = instantiate_classes(config.train_data_connector)
        test_data_connector = instantiate_classes(config.test_data_connector)
        loss = instantiate_classes(config.loss)
        model = instantiate_classes(config.model)
        callbacks = [instantiate_classes(cb) for cb in config.callbacks]

        dg = prints_datagraph_for_config(
            model, train_data_connector, test_data_connector, loss, callbacks
        )

        with open(
            f"{self.gt_config_path}/connection.txt",
            "r",
            encoding="UTF-8",
        ) as f:
            gt_dg = f.read()

        self.assertTrue(content_equal(dg, gt_dg))
