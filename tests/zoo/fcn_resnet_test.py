"""FCN ResNet configs tests."""
import unittest

from .util import content_equal, get_config_for_name


class TestFCNResNetConfig(unittest.TestCase):
    """Tests the content of the provided configs for FCN ResNet."""

    gt_config_path = "tests/vis4d-test-data/config_test/fcn_resnet"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_r50_coco(self) -> None:
        """Test the config."""
        config = get_config_for_name("fcn_resnet.fcn_resnet_coco").to_yaml()

        with open(
            f"{self.gt_config_path}/fcn_resnet_coco.yaml",
            "r",
            encoding="UTF-8",
        ) as f:
            gt_config = f.read()

        self.assertTrue(content_equal(config, gt_config, self.varying_keys))
