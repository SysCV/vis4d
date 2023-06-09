"""BDD100K Faster R-CNN configs tests."""
import unittest

from ..util import content_equal, get_config_for_name


class TestFasterRCNNConfig(unittest.TestCase):
    """Tests the content of the provided configs for Faster R-CNN."""

    config_prefix = "bdd100k.faster_rcnn"
    gt_config_path = "tests/vis4d-test-data/config_test/bdd100k/faster_rcnn"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_r50_fpn_1x(self) -> None:
        """Test the config."""
        config = get_config_for_name(
            f"{self.config_prefix}.faster_rcnn_r50_1x_bdd100k"
        ).to_yaml()

        with open(
            f"{self.gt_config_path}/faster_rcnn_r50_1x_bdd100k.yaml",
            "r",
            encoding="UTF-8",
        ) as f:
            gt_config = f.read()

        self.assertTrue(content_equal(config, gt_config, self.varying_keys))

    def test_r50_fpn_3x(self) -> None:
        """Test the config."""
        config = get_config_for_name(
            f"{self.config_prefix}.faster_rcnn_r50_3x_bdd100k"
        ).to_yaml()

        with open(
            f"{self.gt_config_path}/faster_rcnn_r50_3x_bdd100k.yaml",
            "r",
            encoding="UTF-8",
        ) as f:
            gt_config = f.read()

        self.assertTrue(content_equal(config, gt_config, self.varying_keys))
