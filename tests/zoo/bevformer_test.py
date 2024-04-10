"""BEVFormer configs tests."""

import unittest

from .util import compare_configs


class TestBEVFormerConfig(unittest.TestCase):
    """Tests the content of the provided configs."""

    config_prefix = "bevformer"
    gt_config_path = "tests/vis4d-test-data/config_test/bevformer"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_bevformer_base(self) -> None:
        """Test the config."""
        cfg_gt = f"{self.gt_config_path}/bevformer_base.yaml"

        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.bevformer_base",
                cfg_gt,
                self.varying_keys,
            )
        )

    def test_bevformer_tiny(self) -> None:
        """Test the config."""
        cfg_gt = f"{self.gt_config_path}/bevformer_tiny.yaml"

        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.bevformer_tiny",
                cfg_gt,
                self.varying_keys,
            )
        )

    def test_bevformer_vis(self) -> None:
        """Test the config."""
        cfg_gt = f"{self.gt_config_path}/bevformer_vis.yaml"

        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.bevformer_vis",
                cfg_gt,
                self.varying_keys,
            )
        )
