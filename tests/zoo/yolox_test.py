"""YOLOX configs tests."""
import unittest

from .util import compare_configs


class TestYOLOXConfig(unittest.TestCase):
    """Tests the content of the provided configs for YOLOX."""

    config_prefix = "yolox"
    gt_config_path = "tests/vis4d-test-data/config_test/yolox"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_yolox_coco(self) -> None:
        """Test the config for YOLOX COCO.

        This instantiates the config and compares it to a ground truth.
        """
        compare_configs(
            f"{self.config_prefix}.yolox_coco",
            f"{self.gt_config_path}/yolox_coco.yaml",
            self.varying_keys,
        )
