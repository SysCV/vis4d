"""SHIFT Mask R-CNN configs tests."""

import unittest

from ..util import compare_configs


class TestMaskRCNNConfig(unittest.TestCase):
    """Tests the content of the provided configs for Mask R-CNN."""

    config_prefix = "shift.mask_rcnn"
    gt_config_path = "tests/vis4d-test-data/config_test/shift/mask_rcnn"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_r50_fpn_12e(self) -> None:
        """Test the config."""
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.mask_rcnn_r50_12e_shift",
                f"{self.gt_config_path}/mask_rcnn_r50_12e_shift.yaml",
                self.varying_keys,
            )
        )

    def test_r50_fpn_36e(self) -> None:
        """Test the config."""
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.mask_rcnn_r50_36e_shift",
                f"{self.gt_config_path}/mask_rcnn_r50_36e_shift.yaml",
                self.varying_keys,
            )
        )

    def test_r50_fpn_6e_all_domains(self) -> None:
        """Test the config."""
        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.mask_rcnn_r50_6e_shift_all_domains",
                f"{self.gt_config_path}/mask_rcnn_r50_6e_shift_all_domains.yaml",  # pylint: disable=line-too-long
                self.varying_keys,
            )
        )
