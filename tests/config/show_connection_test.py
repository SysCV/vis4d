"""Show config connection tests."""

import unittest

from tests.util import content_equal, get_test_file
from tests.zoo.util import get_config_for_name
from vis4d.config import instantiate_classes
from vis4d.config.show_connection import prints_datagraph_for_config


class TestShowConfig(unittest.TestCase):
    """Tests the content of the provided configs for Show."""

    def test_show_frcnn(self) -> None:
        """Test the config for faster_rcnn_coco.py.

        This instantiates the config and compares it to a ground truth.
        """
        config = get_config_for_name("faster_rcnn.faster_rcnn_coco")

        train_data_connector = instantiate_classes(config.train_data_connector)
        test_data_connector = instantiate_classes(config.test_data_connector)
        loss = instantiate_classes(config.loss)
        model = instantiate_classes(config.model)

        # Change the data root of evaluator callback to the test data
        config.callbacks[3].init_args.evaluator.init_args.data_root = (
            "tests/vis4d-test-data/coco_test"
        )
        config.callbacks[3].init_args.evaluator.init_args.split = "train"

        callbacks = [instantiate_classes(cb) for cb in config.callbacks]

        dg = prints_datagraph_for_config(
            model, train_data_connector, test_data_connector, loss, callbacks
        )

        with open(get_test_file("connection.txt"), "r", encoding="UTF-8") as f:
            gt_dg = f.read()

        self.assertTrue(content_equal(dg, gt_dg))
