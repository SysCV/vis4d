"""SHIFT Semantic FPN configs tests."""
import unittest

from ..util import compare_configs


class TestSemanticFPNConfig(unittest.TestCase):
    """Tests the content of the provided configs for Semantic-FPN."""

    config_prefix = "shift.semantic_fpn"
    gt_config_path = "tests/vis4d-test-data/config_test/shift/semantic_fpn"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_r50_fpn_160k(self) -> None:
        """Test the config."""
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.semantic_fpn_r50_160k_shift",
                f"{self.gt_config_path}/semantic_fpn_r50_160k_shift.yaml",
                self.varying_keys,
            )
        )

    def test_r50_fpn_40k(self) -> None:
        """Test the config."""
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.semantic_fpn_r50_40k_shift",
                f"{self.gt_config_path}/semantic_fpn_r50_40k_shift.yaml",
                self.varying_keys,
            )
        )

    def test_r50_fpn_160k_all_domains(self) -> None:
        """Test the config."""
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.semantic_fpn_r50_160k_shift_all_domains",
                f"{self.gt_config_path}/semantic_fpn_r50_160k_shift_all_domains.yaml",
                self.varying_keys,
            )
        )

    def test_r50_fpn_40k_all_domains(self) -> None:
        """Test the config."""
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.semantic_fpn_r50_40k_shift_all_domains",
                f"{self.gt_config_path}/semantic_fpn_r50_40k_shift_all_domains.yaml",
                self.varying_keys,
            )
        )
