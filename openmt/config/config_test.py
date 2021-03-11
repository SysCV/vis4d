"""Test cases for parsing config."""
import unittest
from argparse import Namespace

from ..unittest.util import get_test_file
from .config import parse_config, read_config


class TestLoadConfig(unittest.TestCase):
    """Test cases for openmt config parsing."""

    def test_det_yaml(self) -> None:
        """Check detection configuration in yaml format."""
        config = read_config(get_test_file("config-det.yaml"))
        self.assertEqual(config.detection.model_base, "faster-rcnn")
        self.assertEqual(config.solver.base_lr, 0.02)
        self.assertEqual(config.solver.lr_policy, "step")

    def test_det_toml(self) -> None:
        """Check detection configuration in toml format."""
        config = read_config(get_test_file("config-det.toml"))
        self.assertEqual(config.detection.model_base, "faster-rcnn")
        self.assertEqual(config.solver.base_lr, 0.02)
        self.assertEqual(config.solver.lr_policy, "step")

    def test_det_args(self) -> None:
        """Check cmd line argument parsing to launch cfg."""
        args = Namespace(config=get_test_file("config-det.yaml"), num_gpus=2)
        cfg = parse_config(args)
        self.assertEqual(cfg.launch.num_gpus, 2)

    def test_det_notsupported(self) -> None:
        """Check detection configuration in not-supported format."""
        self.assertRaises(NotImplementedError, read_config, "")
