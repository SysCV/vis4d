"""FCN ResNet configs tests."""
import unittest

from .util import compare_configs


class TestFCNResNetConfig(unittest.TestCase):
    """Tests the content of the provided configs for FCN ResNet."""

    config_prefix = "fcn_resnet"
    gt_config_path = "tests/vis4d-test-data/config_test/fcn_resnet"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_r50_coco(self) -> None:
        """Test the config."""
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.fcn_resnet_coco",
                f"{self.gt_config_path}/fcn_resnet_coco.yaml",
                self.varying_keys,
            )
        )
