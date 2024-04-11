"""QDTrack configs tests."""

import unittest

from .util import compare_configs


class TestQDTrackConfig(unittest.TestCase):
    """Tests the content of the provided configs for QDTrack."""

    config_prefix = "qdtrack"
    gt_config_path = "tests/vis4d-test-data/zoo_test/qdtrack"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_frcnn_r50_fpn_augs_1x_bdd100k(self) -> None:
        """Test the config for QDTrack Faster-RCNN.

        This instantiates the config and compares it to a ground truth.
        """
        compare_configs(
            f"{self.config_prefix}.qdtrack_frcnn_r50_fpn_augs_1x_bdd100k",
            f"{self.gt_config_path}/"
            + "qdtrack_frcnn_r50_fpn_augs_1x_bdd100k.yaml",
            self.varying_keys,
        )

    def test_yolox_x_50e_bdd100k(self) -> None:
        """Test the config for QDTrack YOLOX.

        This instantiates the config and compares it to a ground truth.
        """
        compare_configs(
            f"{self.config_prefix}.qdtrack_yolox_x_50e_bdd100k",
            f"{self.gt_config_path}/qdtrack_yolox_x_50e_bdd100k.yaml",
            self.varying_keys,
        )
