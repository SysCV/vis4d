"""BDD100K Semantic FPN configs tests."""

import unittest

from ..util import compare_configs


class TestSemanticFPNConfig(unittest.TestCase):
    """Tests the content of the provided configs for Semantic-FPN."""

    config_prefix = "bdd100k.semantic_fpn"
    gt_config_path = "tests/vis4d-test-data/config_test/bdd100k/semantic_fpn"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_r50_fpn_40k(self) -> None:
        """Test the config."""
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.semantic_fpn_r50_40k_bdd100k",
                f"{self.gt_config_path}/semantic_fpn_r50_40k_bdd100k.yaml",
                self.varying_keys,
            )
        )

    def test_r50_fpn_80k(self) -> None:
        """Test the config."""
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.semantic_fpn_r50_80k_bdd100k",
                f"{self.gt_config_path}/semantic_fpn_r50_80k_bdd100k.yaml",
                self.varying_keys,
            )
        )

    def test_r101_fpn_80k(self) -> None:
        """Test the config."""
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.semantic_fpn_r101_80k_bdd100k",
                f"{self.gt_config_path}/semantic_fpn_r101_80k_bdd100k.yaml",
                self.varying_keys,
            )
        )
