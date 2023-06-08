"""CC-3DT configs tests."""
import unittest

from .util import content_equal, get_config_for_name


class TestCC3DTConfig(unittest.TestCase):
    """Tests the content of the provided configs for Faster-RCNN."""

    gt_config_path = "tests/vis4d-test-data/config_test/cc_3dt"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_frcnn_r50_fpn_nusc(self) -> None:
        """Test the config for cc_3dt_nusc.py.

        This instantiates the config and compares it to a ground truth.
        """
        config = get_config_for_name("cc_3dt.cc_3dt_nusc").to_yaml()

        with open(
            f"{self.gt_config_path}/cc_3dt_nusc.yaml",
            "r",
            encoding="UTF-8",
        ) as f:
            gt_config = f.read()

        self.assertTrue(content_equal(config, gt_config, self.varying_keys))
