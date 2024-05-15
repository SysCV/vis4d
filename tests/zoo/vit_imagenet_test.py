"""ViT configs tests."""

import unittest

from .util import compare_configs


class TestViTConfig(unittest.TestCase):
    """Tests the content of the provided configs for ViT."""

    config_prefix = "vit"
    gt_config_path = "tests/vis4d-test-data/zoo_test/vit"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_small_imagenet(self) -> None:
        """Test the config for vit_small_imagenet.py.

        This instantiates the config and compares it to a ground truth.
        """
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.vit_small_imagenet",
                f"{self.gt_config_path}/vit_small_imagenet.yaml",
                self.varying_keys,
            )
        )

    def test_tiny_imagenet(self) -> None:
        """Test the config for vit_tiny_imagenet.py.

        This instantiates the config and compares it to a ground truth.
        """
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.vit_tiny_imagenet",
                f"{self.gt_config_path}/vit_tiny_imagenet.yaml",
                self.varying_keys,
            )
        )
