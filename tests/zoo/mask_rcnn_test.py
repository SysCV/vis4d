"""Mask-RCNN configs tests."""
import unittest

from .util import content_equal, get_config_for_name


class TestMaskRCNNConfig(unittest.TestCase):
    """Tests the content of the provided configs for Mask-RCNN."""

    gt_config_path = "tests/vis4d-test-data/config_test/mask_rcnn"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_r50_fpn_coco(self) -> None:
        """Test the config."""
        config = get_config_for_name("mask_rcnn.mask_rcnn_coco").to_yaml()

        with open(
            f"{self.gt_config_path}/mask_rcnn_coco.yaml",
            "r",
            encoding="UTF-8",
        ) as f:
            gt_config = f.read()

        self.assertTrue(content_equal(config, gt_config, self.varying_keys))
