"""CC-3DT configs tests."""

import unittest

from .util import compare_configs


class TestCC3DTConfig(unittest.TestCase):
    """Tests the content of the provided configs."""

    config_prefix = "cc_3dt"
    gt_config_path = "tests/vis4d-test-data/zoo_test/cc_3dt"
    varying_keys = ["save_prefix", "output_dir", "version", "timestamp"]

    def test_frcnn_r50_fpn_kf3d_12e_nusc(self) -> None:
        """Test the config."""
        cfg_gt = (
            f"{self.gt_config_path}/cc_3dt_frcnn_r50_fpn_kf3d_12e_nusc.yaml"
        )

        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.cc_3dt_frcnn_r50_fpn_kf3d_12e_nusc",
                cfg_gt,
                self.varying_keys,
            )
        )

    def test_frcnn_r101_fpn_kf3d_24e_nusc(self) -> None:
        """Test the config."""
        cfg_gt = (
            f"{self.gt_config_path}/cc_3dt_frcnn_r101_fpn_kf3d_24e_nusc.yaml"
        )

        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.cc_3dt_frcnn_r101_fpn_kf3d_24e_nusc",
                cfg_gt,
                self.varying_keys,
            )
        )

    def test_frcnn_r101_fpn_pure_det_nusc(self) -> None:
        """Test the config."""
        cfg_gt = (
            f"{self.gt_config_path}/cc_3dt_frcnn_r101_fpn_pure_det_nusc.yaml"
        )

        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.cc_3dt_frcnn_r101_fpn_pure_det_nusc",
                cfg_gt,
                self.varying_keys,
            )
        )

    def test_frcnn_r101_fpn_velo_lstm_24e_nusc(self) -> None:
        """Test the config."""
        cfg_gt = (
            f"{self.gt_config_path}/"
            + "cc_3dt_frcnn_r101_fpn_velo_lstm_24e_nusc.yaml"
        )

        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}."
                + "cc_3dt_frcnn_r101_fpn_velo_lstm_24e_nusc",
                cfg_gt,
                self.varying_keys,
            )
        )

    def test_velo_lstm_frcnn_r101_fpn_100e_nusc(self) -> None:
        """Test the config."""
        cfg_gt = (
            f"{self.gt_config_path}/velo_lstm_frcnn_r101_fpn_100e_nusc.yaml"
        )

        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.velo_lstm_frcnn_r101_fpn_100e_nusc",
                cfg_gt,
                self.varying_keys,
            )
        )

    def test_nusc_vis(self) -> None:
        """Test the config."""
        cfg_gt = f"{self.gt_config_path}/cc_3dt_nusc_vis.yaml"

        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.cc_3dt_nusc_vis",
                cfg_gt,
                self.varying_keys,
            )
        )

    def test_nusc_test(self) -> None:
        """Test the config."""
        cfg_gt = f"{self.gt_config_path}/cc_3dt_nusc_test.yaml"

        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.cc_3dt_nusc_test",
                cfg_gt,
                self.varying_keys,
            )
        )

    def test_bevformer_base_velo_lstm_nusc(self) -> None:
        """Test the config."""
        cfg_gt = (
            f"{self.gt_config_path}/cc_3dt_bevformer_base_velo_lstm_nusc.yaml"
        )

        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.cc_3dt_bevformer_base_velo_lstm_nusc",
                cfg_gt,
                self.varying_keys,
            )
        )

    def test_velo_lstm_bevformer_base_100e_nusc(self) -> None:
        """Test the config."""
        cfg_gt = (
            f"{self.gt_config_path}/velo_lstm_bevformer_base_100e_nusc.yaml"
        )

        self.assertTrue(
            compare_configs(
                f"{self.config_prefix}.velo_lstm_bevformer_base_100e_nusc",
                cfg_gt,
                self.varying_keys,
            )
        )
