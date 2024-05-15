"""Faster-RCNN configs tests."""

import unittest

from .util import compare_configs


class TestFasterRCNNConfig(unittest.TestCase):
    """Tests the content of the provided configs for Faster-RCNN."""

    config_prefix = "faster_rcnn"
    gt_config_path = "tests/vis4d-test-data/zoo_test/faster_rcnn"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_r50_fpn_coco(self) -> None:
        """Test the config for faster_rcnn_coco.py.

        This instantiates the config and compares it to a ground truth.
        """
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.faster_rcnn_coco",
                f"{self.gt_config_path}/faster_rcnn_coco.yaml",
                self.varying_keys,
            )
        )
