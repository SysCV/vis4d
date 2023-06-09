"""QDTrack configs tests."""
import unittest

from .util import content_equal, get_config_for_name


class TestQDTrackConfig(unittest.TestCase):
    """Tests the content of the provided configs for QDTrack."""

    gt_config_path = "tests/vis4d-test-data/config_test/qdtrack"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_r50_fpn_bdd100k(self) -> None:
        """Test the config for QDTrack Faster-RCNN.

        This instantiates the config and compares it to a ground truth.
        """
        config = get_config_for_name("qdtrack.qdtrack_bdd100k").to_yaml()

        with open(
            f"{self.gt_config_path}/qdtrack_bdd100k.yaml",
            "r",
            encoding="UTF-8",
        ) as f:
            gt_config = f.read()

        self.assertTrue(content_equal(config, gt_config, self.varying_keys))
