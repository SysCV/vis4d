"""Test cases for parsing config."""
import unittest

from ..unittest.util import get_test_file
from .config import read_config


class TestLoadConfig(unittest.TestCase):
    """Test cases for systm config parsing."""

    def test_det_yaml(self) -> None:
        """Check detection configuration in yaml format."""
        config = read_config(get_test_file("config_det.yaml"))
        self.assertEqual(config.detection.model_base, "faster-rcnn")
        self.assertEqual(config.solver.base_lr, 0.02)
        self.assertEqual(config.solver.lr_policy, "step")

    def test_det_toml(self) -> None:
        """Check detection configuration in toml format."""
        config = read_config(get_test_file("config_det.toml"))
        self.assertEqual(config.detection.model_base, "faster-rcnn")
        self.assertEqual(config.solver.base_lr, 0.02)
        self.assertEqual(config.solver.lr_policy, "step")
