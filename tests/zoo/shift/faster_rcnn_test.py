"""SHIFT Faster R-CNN configs tests."""
import unittest

from ..util import compare_configs


class TestFasterRCNNConfig(unittest.TestCase):
    """Tests the content of the provided configs for Faster R-CNN."""

    config_prefix = "shift.faster_rcnn"
    gt_config_path = "tests/vis4d-test-data/config_test/shift/faster_rcnn"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_r50_fpn_1x(self) -> None:
        """Test the config."""
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.faster_rcnn_r50_1x_shift",
                f"{self.gt_config_path}/faster_rcnn_r50_1x_shift.yaml",
                self.varying_keys,
            )
        )
