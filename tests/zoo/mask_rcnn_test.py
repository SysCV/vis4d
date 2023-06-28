"""Mask-RCNN configs tests."""
import unittest

from .util import compare_configs


class TestMaskRCNNConfig(unittest.TestCase):
    """Tests the content of the provided configs for Mask-RCNN."""

    config_prefix = "mask_rcnn"
    gt_config_path = "tests/vis4d-test-data/config_test/mask_rcnn"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_r50_fpn_coco(self) -> None:
        """Test the config."""
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.mask_rcnn_coco",
                f"{self.gt_config_path}/mask_rcnn_coco.yaml",
                self.varying_keys,
            )
        )
