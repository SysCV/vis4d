"""BDD100K Faster R-CNN configs tests."""

import unittest

from ..util import compare_configs


class TestFasterRCNNConfig(unittest.TestCase):
    """Tests the content of the provided configs for Faster R-CNN."""

    config_prefix = "bdd100k.faster_rcnn"
    gt_config_path = "tests/vis4d-test-data/config_test/bdd100k/faster_rcnn"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_r50_fpn_1x(self) -> None:
        """Test the config."""
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.faster_rcnn_r50_1x_bdd100k",
                f"{self.gt_config_path}/faster_rcnn_r50_1x_bdd100k.yaml",
                self.varying_keys,
            )
        )

    def test_r50_fpn_3x(self) -> None:
        """Test the config."""
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.faster_rcnn_r50_3x_bdd100k",
                f"{self.gt_config_path}/faster_rcnn_r50_3x_bdd100k.yaml",
                self.varying_keys,
            )
        )
