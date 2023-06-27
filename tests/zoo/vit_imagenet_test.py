"""ViT configs tests."""
import unittest

from .util import content_equal, get_config_for_name


class TestViTConfig(unittest.TestCase):
    """Tests the content of the provided configs for ViT."""

    gt_config_path = "tests/vis4d-test-data/config_test/vit"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_small_imagenet(self) -> None:
        """Test the config for vit_small_imagenet.py.

        This instantiates the config and compares it to a ground truth.
        """
        config = get_config_for_name("vit.vit_small_imagenet").to_yaml()

        with open(
            f"{self.gt_config_path}/vit_small_imagenet.yaml",
            "r",
            encoding="UTF-8",
        ) as f:
            gt_config = f.read()

        self.assertTrue(content_equal(config, gt_config, self.varying_keys))

    def test_tiny_imagenet(self) -> None:
        """Test the config for vit_tiny_imagenet.py.

        This instantiates the config and compares it to a ground truth.
        """
        config = get_config_for_name("vit.vit_tiny_imagenet").to_yaml()

        with open(
            f"{self.gt_config_path}/vit_tiny_imagenet.yaml",
            "r",
            encoding="UTF-8",
        ) as f:
            gt_config = f.read()

        self.assertTrue(content_equal(config, gt_config, self.varying_keys))
