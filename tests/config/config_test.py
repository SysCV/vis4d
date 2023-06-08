"""Test case for experiment config content."""

import importlib
import unittest

from tests.util import content_equal
from vis4d.config import FieldConfigDict


def get_config_for_name(config_name: str) -> FieldConfigDict:
    """Get config for name."""
    module = importlib.import_module("vis4d.zoo." + config_name)
    return module.get_config()


class TestConfigs(unittest.TestCase):
    """Tests the content of the provided configs."""

    gt_config_path = "tests/vis4d-test-data/configs/"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_MaskRcnnConfig(self) -> None:
        """Test the config for MaskRCNN.

        This instantiates the config and compares it to a ground truth.
        """
        self.assertTrue(
            content_equal(
                get_config_for_name("mask_rcnn.mask_rcnn_coco").to_yaml(),
                open(
                    self.gt_config_path + "mask_rcnn_coco.yaml",
                    "r",
                    encoding="UTF-8",
                ).read(),
                self.varying_keys,
            )
        )
