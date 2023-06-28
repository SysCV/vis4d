"""CC-3DT configs tests."""
import unittest

from .util import compare_configs


class TestCC3DTConfig(unittest.TestCase):
    """Tests the content of the provided configs for Faster-RCNN."""

    config_prefix = "cc_3dt"
    gt_config_path = "tests/vis4d-test-data/config_test/cc_3dt"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_frcnn_r50_fpn_nusc(self) -> None:
        """Test the config for cc_3dt_nusc.py.

        This instantiates the config and compares it to a ground truth.
        """
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.cc_3dt_nusc",
                f"{self.gt_config_path}/cc_3dt_nusc.yaml",
                self.varying_keys,
            )
        )
