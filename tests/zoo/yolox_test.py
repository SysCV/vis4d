"""YOLOX configs tests."""
import unittest

from .util import compare_configs


class TestYOLOXConfig(unittest.TestCase):
    """Tests the content of the provided configs for YOLOX."""

    config_prefix = "yolox"
    gt_config_path = "tests/vis4d-test-data/config_test/yolox"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_yolox_tiny_300e_coco(self) -> None:
        """Test the config for YOLOX-t COCO.

        This instantiates the config and compares it to a ground truth.
        """
        compare_configs(
            f"{self.config_prefix}.yolox_tiny_300e_coco",
            f"{self.gt_config_path}/yolox_tiny_300e_coco.yaml",
            self.varying_keys,
        )

    def test_yolox_s_300e_coco(self) -> None:
        """Test the config for YOLOX-s COCO.

        This instantiates the config and compares it to a ground truth.
        """
        compare_configs(
            f"{self.config_prefix}.yolox_s_300e_coco",
            f"{self.gt_config_path}/yolox_s_300e_coco.yaml",
            self.varying_keys,
        )
