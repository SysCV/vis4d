"""RetinaNet configs tests."""

import unittest

from .util import compare_configs


class TestRetinaNetConfig(unittest.TestCase):
    """Tests the content of the provided configs for RetinaNet."""

    config_prefix = "retinanet"
    gt_config_path = "tests/vis4d-test-data/config_test/retinanet"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_r50_fpn_coco(self) -> None:
        """Test the config for faster_rcnn_coco.py.

        This instantiates the config and compares it to a ground truth.
        """
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.retinanet_coco",
                f"{self.gt_config_path}/retinanet_coco.yaml",
                self.varying_keys,
            )
        )
