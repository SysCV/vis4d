"""BDD100K Semantic FPN configs tests."""
import unittest

from ..util import content_equal, get_config_for_name


class TestSemanticFPNConfig(unittest.TestCase):
    """Tests the content of the provided configs for Semantic-FPN."""

    config_prefix = "bdd100k.semantic_fpn"
    gt_config_path = "tests/vis4d-test-data/config_test/bdd100k/semantic_fpn"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_r50_fpn_40k(self) -> None:
        """Test the config."""
        config = get_config_for_name(
            f"{self.config_prefix}.semantic_fpn_r50_40k_bdd100k"
        ).to_yaml()

        with open(
            f"{self.gt_config_path}/semantic_fpn_r50_40k_bdd100k.yaml",
            "r",
            encoding="UTF-8",
        ) as f:
            gt_config = f.read()

        self.assertTrue(content_equal(config, gt_config, self.varying_keys))

    def test_r50_fpn_80k(self) -> None:
        """Test the config."""
        config = get_config_for_name(
            f"{self.config_prefix}.semantic_fpn_r50_80k_bdd100k"
        ).to_yaml()

        with open(
            f"{self.gt_config_path}/semantic_fpn_r50_80k_bdd100k.yaml",
            "r",
            encoding="UTF-8",
        ) as f:
            gt_config = f.read()

        self.assertTrue(content_equal(config, gt_config, self.varying_keys))

    def test_r101_fpn_80k(self) -> None:
        """Test the config."""
        config = get_config_for_name(
            f"{self.config_prefix}.semantic_fpn_r101_80k_bdd100k"
        ).to_yaml()

        with open(
            f"{self.gt_config_path}/semantic_fpn_r101_80k_bdd100k.yaml",
            "r",
            encoding="UTF-8",
        ) as f:
            gt_config = f.read()

        self.assertTrue(content_equal(config, gt_config, self.varying_keys))
